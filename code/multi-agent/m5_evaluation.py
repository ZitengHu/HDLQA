
from __future__ import annotations

import json
import logging
import re
import threading
from typing import Optional

logger = logging.getLogger(__name__)

# TaskCategory 在真实运行时从 pipeline 导入；导入失败时用本地等价定义兜底
try:
    from pipeline import TaskCategory  # type: ignore
except Exception:
    from enum import Enum

    class TaskCategory(str, Enum):  # type: ignore
        CONCEPTUAL   = "conceptual"
        DEBUGGING    = "debugging"
        GENERATION   = "generation"
        OPTIMIZATION = "optimization"


# =============================================================================
# 一、工程师答案 / 外部 baseline LLM 答案的提取（兼容真实数据格式）
# =============================================================================

def extract_engineer_answer(record: dict,
                            accepted_field: str = "accepted_answer") -> Optional[str]:
    """从一条数据集记录中提取工程师 accepted answer 文本。"""
    answers = record.get("answers")
    if isinstance(answers, list):
        for a in answers:
            if not isinstance(a, dict):
                continue
            atype  = str(a.get("type", "")).lower()
            source = str(a.get("source", "")).lower()
            if atype == "accepted" or source == "engineer":
                body = a.get("body")
                if body and str(body).strip():
                    return str(body).strip()
        for a in answers:
            if isinstance(a, dict) and str(a.get("source", "")).lower() != "llm":
                body = a.get("body")
                if body and str(body).strip():
                    return str(body).strip()

    for key in (accepted_field, "answer", "accepted"):
        v = record.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, dict) and v.get("body"):
            return str(v["body"]).strip()
    return None


def extract_baseline_llm_answer(record: dict,
                                prefer_model: Optional[str] = None
                                ) -> tuple[Optional[str], Optional[str]]:
    """提取一个外部独立 LLM 答案（默认不使用，保留以备将来）。"""
    answers = record.get("answers")
    if not isinstance(answers, list):
        return None, None
    llm_answers: dict[str, str] = {}
    for a in answers:
        if not isinstance(a, dict):
            continue
        if str(a.get("source", "")).lower() == "llm" or \
           str(a.get("type", "")).lower().startswith("llm"):
            body = a.get("body")
            if body and str(body).strip():
                mid = a.get("model_id") or a.get("model_name") \
                      or str(a.get("type", "")).replace("llm_", "")
                llm_answers[str(mid)] = str(body).strip()
    if not llm_answers:
        return None, None
    if prefer_model and prefer_model in llm_answers:
        return llm_answers[prefer_model], prefer_model
    metrics = record.get("llm_metrics")
    if isinstance(metrics, dict) and metrics:
        best_model, best_bleu = None, -1.0
        for mid, m in metrics.items():
            if mid in llm_answers and isinstance(m, dict):
                try:
                    bleu = float(m.get("bleu_1", 0))
                except (TypeError, ValueError):
                    bleu = 0.0
                if bleu > best_bleu:
                    best_bleu, best_model = bleu, mid
        if best_model:
            return llm_answers[best_model], best_model
    mid = next(iter(llm_answers))
    return llm_answers[mid], mid


# =============================================================================
# 二、核心抽取
# =============================================================================

_CORE_OPEN  = "⟦CORE⟧"
_CORE_CLOSE = "⟦/CORE⟧"


def _strip_core_markers(text: str) -> str:
    return (text or "").replace(_CORE_OPEN, "").replace(_CORE_CLOSE, "").strip()


def extract_core_by_markers(text: str) -> Optional[str]:
    """从带 ⟦CORE⟧ 标记的文本中抽取核心；无标记返回 None。"""
    if not text or _CORE_OPEN not in text or _CORE_CLOSE not in text:
        return None
    try:
        _, rest = text.split(_CORE_OPEN, 1)
        core, _ = rest.split(_CORE_CLOSE, 1)
        core = core.strip()
        return core or None
    except ValueError:
        return None


def extract_all_cores_by_markers(text: str) -> list[str]:
    """抽取文本中所有 ⟦CORE⟧…⟦/CORE⟧ 段（可能多个核心）。无标记返回 []。"""
    if not text:
        return []
    return [m.strip() for m in re.findall(
        re.escape(_CORE_OPEN) + r"(.*?)" + re.escape(_CORE_CLOSE),
        text, flags=re.DOTALL) if m.strip()]


def _noncore_text(text_with_markers: str) -> str:
    """抽取带标记文本里【非核心(context)】部分的文本。
    非核心 = 全文去掉所有 ⟦CORE⟧…⟦/CORE⟧ 段后剩下的内容。"""
    if not text_with_markers:
        return ""
    stripped = re.sub(
        re.escape(_CORE_OPEN) + r".*?" + re.escape(_CORE_CLOSE),
        "", text_with_markers, flags=re.DOTALL)
    stripped = stripped.replace(_CORE_OPEN, "").replace(_CORE_CLOSE, "")
    return stripped.strip()


def _noncore_length(text_with_markers: str) -> int:
    """估算带标记文本里【非核心】部分的字符数（用于 M4 长度基准）。"""
    return len(_noncore_text(text_with_markers))


# 各类别「核心」的定义，用于（a）M2草稿抽核心 (b)向 judge 说明只比什么
_CORE_FOCUS: dict[str, str] = {
    TaskCategory.GENERATION.value:
        "the CODE that implements the requirement (the implementation approach embodied in the code)",
    TaskCategory.DEBUGGING.value:
        "the DEBUG/FIX approach (the diagnosed cause and the fix strategy)",
    TaskCategory.CONCEPTUAL.value:
        "the EXPLANATION of the concept/syntax/semantics (the technical reasoning)",
    TaskCategory.OPTIMIZATION.value:
        "the OPTIMIZATION method/approach (the concrete optimization idea)",
}


# =============================================================================
# 三、M5 Judge —— 合并三指标的单次调用（只比核心 + baseline口径A）
#     仅在 enable_engineer_eval=True 时调用。
# =============================================================================

_M5_SYSTEM = (
    "You are a senior Verilog/SystemVerilog/VHDL answer evaluator.\n"
    "You compare candidate HDL answers against a reference answer written by the engineer who "
    "asked the question (the ACCEPTED answer). You perform THREE tasks in one pass and return a "
    "single JSON object.\n\n"

    "CRITICAL — CONSISTENCY COMPARES ONLY THE CORE:\n"
    "For consistency, you ONLY compare the CORE of each answer, defined by the task category:\n"
    "  generation   -> ONLY the code / implementation approach\n"
    "  debugging    -> ONLY the debug/fix approach (cause + fix strategy)\n"
    "  conceptual   -> ONLY the explanation of the concept/syntax/semantics\n"
    "  optimization -> ONLY the optimization method/approach\n"
    "Ignore everything outside the core: intro, restated question, general background, closing "
    "remarks, usage notes, testbench, and non-core context.\n\n"

    "GUIDING PRINCIPLES:\n"
    "1. Compare the core technical APPROACH/CONCLUSION, never wording or verbosity.\n"
    "2. Saying more OUTSIDE the core is NOT penalised. But saying more INSIDE the core can hurt "
    "if the extra core approaches are redundant, derivative, generic, speculative, or unsupported.\n"
    "3. A core may contain multiple distinct approaches. Count them carefully.\n"
    "   For total_approaches, count every distinct core-level proposed approach, diagnosis, fix, "
    "or solution branch that the answer presents as part of the solution.\n"
    "4. For consistent_approaches, count ONLY approaches that BOTH:\n"
    "     (a) agree with the engineer's core answer in substance; AND\n"
    "     (b) are independently core-worthy: specific, useful, non-derivative, non-generic, and "
    "not merely a wrapper/parameter/scope variant of another approach.\n"
    "   Do NOT count the following as consistent_approaches even if they are not strictly wrong:\n"
    "     - generic checklist advice;\n"
    "     - low-probability speculative long-shots;\n"
    "     - wrapper-only variants of an already counted solution;\n"
    "     - parameter-only or scope-only variants of the same command/API/mechanism;\n"
    "     - redundant weaker duplicates of another approach.\n"
    "5. For conflicting_approaches, count approaches that contradict the engineer, are technically "
    "wrong, non-synthesizable when synthesis is required, or diagnose the wrong root cause.\n"
    "6. Verdict rule:\n"
    "     - All core-worthy approaches agree with the engineer, with no wrong approaches -> consistent\n"
    "     - Some core-worthy approach agrees, but wrong/conflicting or materially unsupported core "
    "content remains -> partially_consistent\n"
    "     - All core approaches contradict the engineer -> contradictory\n"
    "   Redundant/generic/speculative approaches may leave the verdict as consistent if not wrong, "
    "but they MUST reduce precision because they count in total_approaches but not in "
    "consistent_approaches.\n"
    "7. A single-approach core has total_approaches = 1. Its consistent_approaches is 1 only if "
    "that one approach agrees with the engineer AND is independently core-worthy.\n"
    "8. generation: cores can be functionally_equivalent with different implementations if "
    "interface and observable behaviour match.\n"

    "Output ONLY valid JSON — no markdown fences, no preamble, no trailing text."
)

def _build_m5_prompt(req,
                     engineer_answer: str,
                     final_core: str,
                     m2_full: str,
                     m2_core_hint: Optional[str],
                     deleted_blocks: list[dict],
                     category: str,
                     is_generation: bool) -> str:
    """构造合并三指标的 judge prompt（只比核心，baseline口径A=M2核心 vs M3核心）。"""

    verdict_enum = ("consistent | partially_consistent | contradictory | not_applicable"
                    + (" | functionally_equivalent" if is_generation else ""))

    core_focus = _CORE_FOCUS.get(category, "the key technical approach that solves the request")

    # 删除块清单
    if deleted_blocks:
        del_lines = []
        for i, b in enumerate(deleted_blocks):
            ctype   = b.get("content_type", "other")
            content = (b.get("content") or "").strip()
            del_lines.append(f'[block {i}] content_type="{ctype}"\n{content}')
        deleted_block_text = "\n\n".join(del_lines)
    else:
        deleted_block_text = "(no blocks were deleted by M4)"

    # M2 核心：若已有标记抽取结果则直接给，否则给整段并要 judge 自己按 core_focus 抽
    if m2_core_hint:
        m2_section = (
            f"<m2_core source=\"M2_draft_core\">\n{m2_core_hint}\n</m2_core>"
        )
        m2_instruction = "The M2 core is already provided above."
    else:
        m2_section = (
            f"<m2_full source=\"M2_draft_full\">\n{m2_full}\n</m2_full>"
        )
        m2_instruction = (
            f"The M2 draft is given in full (no core markers). First mentally EXTRACT its CORE — "
            f"namely {core_focus} — then compare ONLY that extracted core."
        )

    prompt = f"""Evaluate the following HDL answers against the engineer's accepted answer.

Task category : {category}
For THIS category, the CORE to compare is: {core_focus}

User question : {req.raw_question[:1500]}
User intent   : {req.user_intent}

<engineer_answer>
{engineer_answer}
</engineer_answer>

<final_core source="M3_core_after_pruning">
{final_core}
</final_core>

{m2_section}

The following content blocks were DELETED by the M4 screener (non-core, removed from the final
answer). These are NOT part of the consistency comparison; they are only for the deletion analysis:
<deleted_blocks>
{deleted_block_text}
</deleted_blocks>

Return ONLY a JSON object with EXACTLY this schema:

{{
  "consistency": {{
    "m3_core_vs_engineer": {{
      "verdict": "<{verdict_enum}>",
      "total_approaches": <int: count by reading ONLY the M3 final_core text; M3 is post-pruning so usually FEWER than M2; do NOT reuse M2's count; 1 if single approach>,
      "consistent_approaches": <int: how many of them agree with the engineer's approach>,
      "conflicting_approaches": <int: how many of them contradict / are wrong vs the engineer's>,
      "matched_points": ["..."],
      "missing_points": ["..."],
      "conflicting_points": ["..."],
      "reason": "<one sentence>"
    }},
    "m2_core_vs_engineer": {{
      "verdict": "<{verdict_enum}>",
      "total_approaches": <int: how many distinct solution approaches the M2 core contains (1 if single)>,
      "consistent_approaches": <int: how many of them agree with the engineer's approach>,
      "conflicting_approaches": <int: how many of them contradict / are wrong vs the engineer's>,
      "matched_points": ["..."],
      "missing_points": ["..."],
      "conflicting_points": ["..."],
      "reason": "<one sentence>"
    }}
  }},
  "engineer_noncore_summary": "<one or two sentences: what, if anything, the engineer answer contains OUTSIDE its core (background, restated question, setup, closing remarks, asides). If the engineer answer is essentially all-core with no non-core part, say so explicitly.>",
  "deletion_analysis": {{
    "per_block": [
      {{
        "block_idx": <int, matching [block N] above>,
        "information_class": "<redundant_repetition | generic_background | relevant_supplement | core_point>",
        "in_engineer_noncore": "<present | absent>",
        "reason": "<one sentence>"
      }}
    ]
  }},
  "coverage": {{
    "key_points": [
      {{
        "point": "<short technical claim from the engineer's CORE>",
        "in_m2_core": <true|false>,
        "in_m3_core": <true|false>
      }}
    ]
  }}
}}

TASK 1 — CONSISTENCY (CORE-ONLY, baseline = M2 core):
- Extract the CORE of the engineer answer ({core_focus}).
- m3_core_vs_engineer: compare the provided <final_core> (M3 core, after pruning) against the
  engineer's core.
- m2_core_vs_engineer: {m2_instruction} Compare the M2 core against the engineer's core.
- If a core contains MULTIPLE alternative solution approaches, count them: total_approaches =
  how many distinct approaches; consistent_approaches = how many agree with the engineer;
  conflicting_approaches = how many contradict / are wrong. Apply the multi-approach verdict rule
  from the GUIDING PRINCIPLES (a wrong approach mixed in -> partially_consistent or worse).
- A single-approach core has total_approaches = 1 and consistent_approaches = 1 (if it agrees) or
  0 (if it contradicts).
- CRITICAL — count M2 and M3 INDEPENDENTLY by actually reading each core's own text:
  The M3 core is the result AFTER pruning, so it usually contains FEWER approaches than the M2
  core. Count total_approaches for the M3 core by inspecting ONLY the <final_core> text you were
  given — do NOT carry over or reuse M2's approach count. It is normal and expected for
  M3 total_approaches < M2 total_approaches. If the M3 core text shows only one approach, its
  total_approaches MUST be 1, regardless of how many M2 had.
- Follow the GUIDING PRINCIPLES. Compare ONLY cores. Use not_applicable only if the engineer's
  answer has no extractable core of this type.

TASK 2 — DELETION ANALYSIS (literal comparison against the engineer's NON-CORE part):
- First, identify the engineer answer's NON-CORE part: everything OUTSIDE its core ({core_focus}) —
  i.e. background, restated question, problem setup, asides, closing remarks, generic theory.
  Summarise it in engineer_noncore_summary. If the engineer answer is essentially all core with no
  non-core part, state that.
- Then, for EACH deleted block, decide in_engineer_noncore:
    present -> the same information (in substance) appears in the engineer answer's NON-CORE part.
    absent  -> that information is NOT in the engineer answer's non-core part (the engineer's
               non-core part does not contain it, or the engineer has no non-core part at all).
  This is a LITERAL containment check against the engineer's non-core text — do NOT reason about
  whether the engineer's conclusion depends on it.

TASK 3 — COVERAGE (engineer CORE key points across M2 core vs M3 core):
- Extract the engineer's CORE key technical points (typically 2-6).
- For each point, mark whether it appears (in substance) in the M2 core and in the M3 core."""

    return prompt


def _verdict_to_score(verdict: str) -> Optional[int]:
    v = (verdict or "").strip().lower()
    return {
        "consistent": 3,
        "functionally_equivalent": 3,
        "partially_consistent": 2,
        "contradictory": 1,
        "not_applicable": None,
    }.get(v, None)


def _compute_delta(m3_v: str, m2_v: str) -> str:
    """M3核心 相对 M2核心 的一致性变化方向（口径A）。"""
    s3, s2 = _verdict_to_score(m3_v), _verdict_to_score(m2_v)
    if s3 is None or s2 is None:
        return "unchanged"
    if s3 > s2:
        return "improved"
    if s3 < s2:
        return "regressed"
    return "unchanged"


# =============================================================================
# 三'、M3 核心质量评分（绝对质量，不看工程师答案；与数量/篇幅解耦）
#       —— 始终计算，不受 enable_engineer_eval 影响。
# =============================================================================

_QUALITY_SYSTEM = (
    "You are a senior Verilog/SystemVerilog/VHDL reviewer scoring the CORE SELECTION QUALITY of "
    "an answer. You judge the core ON ITS OWN TECHNICAL MERITS — you are NOT given and must NOT "
    "assume any reference/engineer answer.\n"
    "\n"
    "This score is not just correctness. It measures whether the remaining core is technically "
    "correct, usable, focused, and free of low-value core leftovers.\n"
    "\n"
    "Do NOT penalise shortness, terseness, missing explanation, or different wording. However, DO "
    "penalise extra material that is incorrectly left inside the core, including redundant variants, "
    "wrapper-only alternatives, parameter-only/scope-only variants, generic checklist advice, and "
    "speculative long-shot diagnoses. These are core-selection defects even when they are not "
    "strictly false.\n"
    "\n"
    "Output ONLY valid JSON — no markdown fences, no preamble, no trailing text."
)

# 单分量表（1-5），多维度仅作评判时的考量点，不分别输出
_QUALITY_RUBRIC = (
    "Give a SINGLE integer core-selection-quality score 1-5 for EACH core, considering "
    "holistically:\n"
    "  - Technical correctness: HDL semantics/logic are correct; no wrong assertions or wrong "
    "diagnosis.\n"
    "  - Synthesizability / usability: the code or approach is synthesizable and practically usable.\n"
    "  - Core focus: the core contains only independently useful solution approaches, diagnoses, "
    "or implementation mechanisms.\n"
    "  - No low-value leftovers: the core does NOT still contain redundant derivative variants, "
    "wrapper-only alternatives, parameter-only/scope-only variants, generic checklist items, or "
    "low-probability speculative long-shots.\n"
    "\n"
    "Scale:\n"
    "  5 = clean core: all remaining core content is correct, usable, specific, independently "
    "valuable, and free of wrong, redundant, generic, or speculative leftovers.\n"
    "  4 = technically correct and usable, with only one negligible low-value leftover or minor "
    "focus issue.\n"
    "  3 = mostly correct, but the core still contains noticeable redundant/generic/speculative "
    "material, or a derivative variant that should have been merged/removed.\n"
    "  2 = a wrong/non-synthesizable approach remains, OR multiple low-value leftovers substantially "
    "dilute the core.\n"
    "  1 = the core contains a substantive technical error or the core selection essentially failed.\n"
    "\n"
    "Important distinctions:\n"
    "  - Do NOT lower the score merely because the core is short, terse, or lists fewer options.\n"
    "  - DO lower the score if the core lists more options but some are not independently core-worthy.\n"
    "  - A focused answer with fewer correct core approaches should score higher than a bloated core "
    "that mixes correct approaches with redundant wrappers, generic checks, or weak speculation.\n"
    "  - Do NOT penalize a core merely because it has fewer approaches if it preserves the necessary technical solution."
)

def _build_quality_prompt(req,
                          m3_core: str,
                          m2_full: str,
                          category: str) -> str:
    core_focus = _CORE_FOCUS.get(category, "the key technical approach that solves the request")
    return f"""Score the QUALITY of two cores for the same Verilog/HDL question, independently.

Task category : {category}
For THIS category, the CORE is: {core_focus}

User question : {req.raw_question[:1200]}
User intent   : {req.user_intent}

<m3_core source="M3_core_after_pruning">
{m3_core}
</m3_core>

<m2_full source="M2_draft_full_no_markers">
{m2_full}
</m2_full>

{_QUALITY_RUBRIC}

Instructions:
- m3_core_quality: read ONLY the <m3_core> text and score the quality of its core ({core_focus}).
- m2_core_quality: mentally EXTRACT the core ({core_focus}) from the <m2_full> draft, then score
  THAT extracted core. The M2 draft is BEFORE pruning, so its extracted core may still contain
  wrong, redundant, generic, speculative, wrapper-only, or parameter/scope-only approaches that M3
  is supposed to remove.
- Score the two cores INDEPENDENTLY by reading each one's own content. It is expected that the M3
  core (after pruning) scores > the M2 core when M3 removed wrong, redundant, generic,
  speculative, wrapper-only, or parameter/scope-only core leftovers; but if the M2 core was already
  clean, the scores can be equal.
- Remember: do NOT use any external/engineer answer; judge absolute core-selection quality only.

Return ONLY a JSON object with EXACTLY this schema:
{{
  "m3_core_quality": {{"score": <int 1-5>, "reason": "<one sentence naming the decisive technical factor>"}},
  "m2_core_quality": {{"score": <int 1-5>, "reason": "<one sentence>"}}
}}"""


def _run_quality_judge(req, m3_core: str, m2_full: str, category: str,
                       judge_model_cfg, judge_temperature: float,
                       raw_llm_fn, strip_fences_fn) -> Optional[dict]:
    """独立的一次 judge 调用，仅做 M3/M2 核心绝对质量评分（不看工程师答案）。"""
    prompt = _build_quality_prompt(req, (m3_core or "")[:6000],
                                   (m2_full or "")[:6000], category)
    for temp in (judge_temperature, 0.0):
        try:
            resp = raw_llm_fn(prompt, system=_QUALITY_SYSTEM,
                              temperature=temp, model_cfg=judge_model_cfg)
            data = json.loads(strip_fences_fn(resp))
            if isinstance(data, dict) and "m3_core_quality" in data:
                return data
        except Exception as exc:
            logger.warning("M5 quality judge failed: %s", exc)
    return None


def _clip_score(v) -> Optional[int]:
    try:
        s = int(round(float(v)))
    except (TypeError, ValueError):
        return None
    return max(1, min(5, s))


def _quality_delta(m3_s: Optional[int], m2_s: Optional[int]) -> str:
    if m3_s is None or m2_s is None:
        return "unchanged"
    if m3_s > m2_s:
        return "improved"
    if m3_s < m2_s:
        return "regressed"
    return "unchanged"


# =============================================================================
# 三''、M4 context 效率分（绝对 LLM-as-judge，不看工程师答案）—— 改造
#        对比【M3 分离好的 context】 vs 【M4 精炼好的 context】，各打一个 1-5 合成分。
#        合成分的两个子维度合一个总分:
#          - 信息富度/有用性 (informativeness / usefulness)
#          - 简洁度          (conciseness)
#        与 M3 核心质量分(m2核心→m3核心)对称：M3 管核心前后质量，M4 管 context 前后质量。
#        始终计算（不依赖工程师答案），不受 enable_engineer_eval 影响。
#        无 context（M3 context 与 M4 context 都为空）时跳过打分。
# =============================================================================

_M4_CTXQUAL_SYSTEM = (
    "You are a senior technical editor scoring the EFFICIENCY of the NON-CORE (context) part of an "
    "HDL answer. You are NOT given any reference/engineer answer and must NOT assume one. You judge "
    "the context ON ITS OWN MERITS, never the core/code (another stage owns that — ignore code).\n"
    "Give ONE integer score 1-5 for CONTEXT EFFICIENCY: the remaining context should be necessary, "
    "directly useful, concise, and non-redundant. Do NOT reward context merely for being longer, "
    "more exhaustive, or more explanatory. A short or empty context can score highly when the core is "
    "self-contained and no extra explanation is needed.\n"
    "Output ONLY valid JSON — no markdown fences, no preamble, no trailing text."
)

_M4_CTXQUAL_RUBRIC = (
    "Single integer score 1-5 for context efficiency:\n"
    "  5 = minimal, necessary, directly useful, and non-redundant; no filler or generic recap.\n"
    "  4 = mostly useful and concise, with only minor extra explanation.\n"
    "  3 = some useful context, but noticeable redundancy, background, recap, or unnecessary detail.\n"
    "  2 = bloated or weakly related context; generic background/checklists/recaps dilute usefulness.\n"
    "  1 = mostly filler, boilerplate, restatement, or distracting context.\n"
    "If the context is EMPTY, this is not automatically bad: a self-contained core may need no context. "
    "Only lower the score if missing context makes the answer materially less usable."
)


def _build_m4_ctxqual_prompt(req, category: str,
                             m3_context: str, m4_context: str) -> str:
    core_focus = _CORE_FOCUS.get(category, "the key technical approach that solves the request")
    m3c = (m3_context or "").strip() or "(empty — no context at this stage)"
    m4c = (m4_context or "").strip() or "(empty — no context remains after M4)"
    return f"""Score the EFFICIENCY of the NON-CORE (context) part of an HDL answer, at two stages, independently.

Task category : {category}
The CORE (which you must IGNORE — do not score it) is: {core_focus}
You score ONLY the context (everything that is NOT the core), at each stage, as context efficiency: necessary, concise, directly useful, and non-redundant.

User question : {req.raw_question[:1200]}

<m3_context source="context_after_M3_separation_before_M4">
{m3c}
</m3_context>

<m4_context source="context_after_M4_refinement">
{m4c}
</m4_context>

{_M4_CTXQUAL_RUBRIC}

Instructions:
- m3_context_quality: score the context as it stood AFTER M3 separated it but BEFORE M4 refined it.
- m4_context_quality: score the context AFTER M4 refined it (deletions + compressions applied).
- Score the two INDEPENDENTLY by reading each one's own text. It is expected that M4 context
  scores >= M3 context when M4 removed water / recap and tightened wording; if M3 context was
  already tight, scores can be equal. Do NOT lower M4 just because it is shorter or less exhaustive.
- If a stage's context is empty, set its score to null.
- Judge ONLY context efficiency; never score the core/code. Do not reward verbose background or broad explanations.

Return ONLY a JSON object with EXACTLY this schema:
{{
  "m3_context_quality": {{"score": <int 1-5, or null if empty>, "reason": "<one sentence>"}},
  "m4_context_quality": {{"score": <int 1-5, or null if empty>, "reason": "<one sentence naming the decisive factor>"}}
}}"""


def _run_m4_ctxqual_judge(req, category: str,
                          m3_context: str, m4_context: str,
                          judge_model_cfg, judge_temperature: float,
                          raw_llm_fn, strip_fences_fn) -> Optional[dict]:
    """一次 judge 调用：M4 context 效率分（M3 context vs M4 context，各一个合成分）。
    两侧 context 都为空时返回 None（跳过）。"""
    if not (m3_context or "").strip() and not (m4_context or "").strip():
        return None
    prompt = _build_m4_ctxqual_prompt(req, category,
                                      (m3_context or "")[:6000],
                                      (m4_context or "")[:6000])
    for temp in (judge_temperature, 0.0):
        try:
            resp = raw_llm_fn(prompt, system=_M4_CTXQUAL_SYSTEM,
                              temperature=temp, model_cfg=judge_model_cfg)
            data = json.loads(strip_fences_fn(resp))
            if isinstance(data, dict) and "m4_context_quality" in data:
                return data
        except Exception as exc:
            logger.warning("M5 M4-context-quality judge failed: %s", exc)
    return None


# =============================================================================
# 三'''、M4 长度指标（客观，不调 LLM）—— 新增
#         只针对 M4 处理过的 context，算缩减率。
# =============================================================================

def _compute_m4_length(m3_output: str,
                       deleted_blocks: list[dict],
                       compressed_blocks: Optional[list[dict]]) -> dict:
    """计算 M4 对 context 的长度缩减（客观）。

    返回:
      {
        "noncore_chars_before": int,   # M3 输出里非核心总字符（缩减基准）
        "deleted_chars": int,          # 删除块总字符
        "compressed_before": int,      # 压缩块原始总字符（M4 落地后才有；当前多为 0）
        "compressed_after": int,       # 压缩块压后总字符
        "removed_chars": int,          # 实际去除 = 删除全长 + 压缩缩减量
        "reduction_ratio": float|None, # removed / noncore_before
      }
    """
    noncore_before = _noncore_length(m3_output or "")

    deleted_chars = sum(len((b.get("content") or "").strip()) for b in (deleted_blocks or []))

    comp_before = comp_after = 0
    for b in (compressed_blocks or []):
        comp_before += len((b.get("before") or b.get("content") or "").strip())
        comp_after  += len((b.get("after") or "").strip())

    removed = deleted_chars + max(0, comp_before - comp_after)
    ratio = (removed / noncore_before) if noncore_before > 0 else None

    return {
        "noncore_chars_before": noncore_before,
        "deleted_chars": deleted_chars,
        "compressed_before": comp_before,
        "compressed_after": comp_after,
        "removed_chars": removed,
        "reduction_ratio": ratio,
    }


# =============================================================================
# 四、M3 是否实际剔除过核心 —— 分层判断
# =============================================================================

def _record_is_eliminated(r: dict) -> bool:
    """兼容不同 M3 feedback 字段格式，判断单条候选记录是否表示被淘汰。"""
    if not isinstance(r, dict):
        return False
    if r.get("passed") is False:
        return True
    decision = str(r.get("decision") or r.get("verdict") or r.get("status") or "").strip().lower()
    if decision in {"eliminate", "eliminated", "removed", "pruned", "reject", "rejected", "drop", "dropped"}:
        return True
    return False


def _latest_m3_feedback_records(m3_feedback: list[dict] | None) -> list[dict]:
    """取最后一次 M3 尝试的记录；无 attempt 字段时默认 attempt=1。"""
    if not m3_feedback:
        return []
    latest = max((r.get("attempt", 1) for r in m3_feedback if isinstance(r, dict)), default=1)
    return [r for r in m3_feedback if isinstance(r, dict) and r.get("attempt", 1) == latest]


def m3_pruned_core(m3_feedback: list[dict] | None) -> bool:
    """判断该问题是否被 M3 实际剔除过核心候选。

    分层①(pruned)= M3 至少淘汰过一个 core candidate。
    分层②(untouched)= M3 没有淘汰核心，或只有单一核心直接透传。

    注意：旧版曾把「拆成 >=2 个核心候选」当作 pruned；这里改回真正的
    「实际淘汰过候选」，更适合衡量 M3 pruning 的有效性。
    """
    return any(_record_is_eliminated(r) for r in _latest_m3_feedback_records(m3_feedback))


def m3_multi_candidate_core(m3_feedback: list[dict] | None) -> bool:
    """辅助分层：是否被 M3 拆成多个核心候选。当前聚合表不用它，但保留给下游分析。"""
    return len(_latest_m3_feedback_records(m3_feedback)) >= 2


def _core_pruning_stats(m3_feedback: list[dict] | None) -> dict:
    """不依赖工程师答案的 M3 核心淘汰统计。

    split_candidates: 最新一次 M3 产生/审核的核心候选数
    kept_candidates:  最终保留数
    eliminated_candidates: 淘汰数
    elimination_ratio: 淘汰数 / 候选数
    """
    recs = _latest_m3_feedback_records(m3_feedback)
    if not recs:
        return {
            "split_candidates": None,
            "kept_candidates": None,
            "eliminated_candidates": None,
            "elimination_ratio": None,
        }

    total = len(recs)
    eliminated = sum(1 for r in recs if _record_is_eliminated(r))
    kept = max(0, total - eliminated)
    return {
        "split_candidates": total,
        "kept_candidates": kept,
        "eliminated_candidates": eliminated,
        "elimination_ratio": (eliminated / total) if total else None,
    }

# =============================================================================
# 五、后处理：规整 judge 输出 + 派生统计
# =============================================================================

def _postprocess_m5(raw: Optional[dict],
                    deleted_blocks: list[dict],
                    m3_pruned: bool,
                    quality: Optional[dict] = None,
                    m4_concision: Optional[dict] = None,
                    m4_length: Optional[dict] = None,
                    engineer_eval: bool = True) -> dict:
    """规整 judge 输出。

    engineer_eval=False 时 raw 可为 None（未调用工程师对照 judge）：
    跳过一致性 / 纯度 / 覆盖率 / 删除精度，只保留不依赖工程师答案的指标。
    """
    out: dict = {}
    raw = raw or {}

    # ===== 依赖工程师答案的部分（仅 engineer_eval=True 时填充） =====
    if engineer_eval:
        # ---- 指标1：一致性（口径A：M3核心 vs M2核心） ----
        cons = raw.get("consistency", {}) or {}
        m3_block = cons.get("m3_core_vs_engineer", {}) or {}
        m2_block = cons.get("m2_core_vs_engineer", {}) or {}

        # ---- 精确度（一致方案数 / 该核心总方案数）----
        def _precision(block: dict) -> Optional[float]:
            try:
                total = int(block.get("total_approaches"))
                cons_n = int(block.get("consistent_approaches"))
            except (TypeError, ValueError):
                return None
            if total <= 0:
                return None
            return max(0.0, min(1.0, cons_n / total))

        m3_prec = _precision(m3_block)
        m2_prec = _precision(m2_block)

        # 防护栏：M3 是删减后的核心，其方案数不应多于 M2。
        purity_count_unreliable = False
        try:
            m2_total = int(m2_block.get("total_approaches"))
            m3_total = int(m3_block.get("total_approaches"))
            if m3_total > m2_total:
                purity_count_unreliable = True
        except (TypeError, ValueError):
            pass

        if purity_count_unreliable:
            prec_delta = "unreliable"
        elif m3_prec is not None and m2_prec is not None:
            if m3_prec > m2_prec + 1e-9:
                prec_delta = "improved"
            elif m3_prec < m2_prec - 1e-9:
                prec_delta = "regressed"
            else:
                prec_delta = "unchanged"
        else:
            prec_delta = "unchanged"

        out["consistency"] = {
            "m3_core_vs_engineer": m3_block,
            "m2_core_vs_engineer": m2_block,
            "delta": _compute_delta(m3_block.get("verdict", ""),
                                    m2_block.get("verdict", "")),
            "m3_pruned_core": m3_pruned,    # 分层标记
            "precision": {
                "m2_core": m2_prec,
                "m3_core": m3_prec,
                "delta": prec_delta,
                "count_unreliable": purity_count_unreliable,
            },
        }

        # ---- 指标2：删除分析（字面对比工程师非核心部分） ----
        per_block_raw = (raw.get("deletion_analysis", {}) or {}).get("per_block", []) or []
        by_idx = {}
        for pb in per_block_raw:
            try:
                by_idx[int(pb.get("block_idx"))] = pb
            except (TypeError, ValueError):
                continue
        per_block_out, tn, fn = [], 0, 0
        by_class = {"redundant_repetition": 0, "generic_background": 0,
                    "relevant_supplement": 0, "core_point": 0}
        for i, blk in enumerate(deleted_blocks):
            pb = by_idx.get(i, {})
            info_class = (pb.get("information_class") or "relevant_supplement").strip()
            in_noncore = (pb.get("in_engineer_noncore") or "absent").strip().lower()
            if info_class in by_class:
                by_class[info_class] += 1
            if in_noncore == "present":
                fn += 1     # 工程师非核心里有 -> 误删
            else:
                tn += 1     # 工程师非核心里没有 -> 删得对
            per_block_out.append({
                "block_idx": i,
                "content_type": blk.get("content_type", "other"),
                "content_preview": (blk.get("content") or "").replace("\n", " ")[:80],
                "information_class": info_class,
                "in_engineer_noncore": in_noncore,
                "reason": (pb.get("reason") or "").strip(),
            })
        total_deleted = len(deleted_blocks)
        out["deletion_analysis"] = {
            "engineer_noncore_summary": (raw.get("engineer_noncore_summary") or "").strip(),
            "per_block": per_block_out,
            "summary": {
                "total_deleted": total_deleted,
                "true_negative": tn,      # absent —— 正确删除
                "false_negative": fn,     # present —— 误删（工程师非核心里有）
                "precision": (tn / total_deleted) if total_deleted else None,
                "by_class": by_class,
            },
        }

        # ---- 指标3：覆盖率（M2核心 vs M3核心） ----
        kp_raw = (raw.get("coverage", {}) or {}).get("key_points", []) or []
        kp_out = []
        total = cov_m2 = cov_m3 = 0
        recovered = lost = 0   # M3 相对 M2 的要点变化
        for kp in kp_raw:
            in_m2 = bool(kp.get("in_m2_core"))
            in_m3 = bool(kp.get("in_m3_core"))
            total += 1
            cov_m2 += int(in_m2)
            cov_m3 += int(in_m3)
            if in_m3 and not in_m2:
                recovered += 1     # M3 补回了 M2 没有的工程师要点
            if in_m2 and not in_m3:
                lost += 1          # M3 丢了 M2 本有的工程师要点
            kp_out.append({"point": (kp.get("point") or "").strip(),
                           "in_m2_core": in_m2, "in_m3_core": in_m3})
        out["coverage"] = {
            "key_points": kp_out,
            "summary": {
                "total_points": total,
                "covered_in_m2_core": cov_m2,
                "covered_in_m3_core": cov_m3,
                "recall_m2_core": (cov_m2 / total) if total else None,
                "recall_m3_core": (cov_m3 / total) if total else None,
                "points_recovered_by_m3": recovered,
                "points_lost_by_m3": lost,
            },
        }
    else:
        # 工程师对照关闭：保留分层标记，便于聚合按层归类
        out["consistency"] = {"m3_pruned_core": m3_pruned, "engineer_eval": False}

    # ===== 不依赖工程师答案的部分（始终填充） =====

    # ---- 指标4：M3 核心质量评分（绝对质量） ----
    q = quality or {}
    m3q = _clip_score((q.get("m3_core_quality") or {}).get("score"))
    m2q = _clip_score((q.get("m2_core_quality") or {}).get("score"))
    out["core_quality"] = {
        "m2_core": {
            "score": m2q,
            "reason": ((q.get("m2_core_quality") or {}).get("reason") or "").strip(),
        },
        "m3_core": {
            "score": m3q,
            "reason": ((q.get("m3_core_quality") or {}).get("reason") or "").strip(),
        },
        "delta": _quality_delta(m3q, m2q),
    }

    # ---- 指标5：M4 长度（客观） ----
    if m4_length is not None:
        out["m4_length"] = m4_length

    # ---- 指标6：M4 context 效率分（绝对：必要性/有用性 + 简洁度 + 非冗余，合成一分；M3ctx→M4ctx） ----
    mc = m4_concision or {}   # 形参名沿用 m4_concision，承载 ctxqual 结果
    m3c = _clip_score((mc.get("m3_context_quality") or {}).get("score"))
    m4c = _clip_score((mc.get("m4_context_quality") or {}).get("score"))
    out["m4_context_quality"] = {
        "m3_context": {
            "score": m3c,
            "reason": ((mc.get("m3_context_quality") or {}).get("reason") or "").strip(),
        },
        "m4_context": {
            "score": m4c,
            "reason": ((mc.get("m4_context_quality") or {}).get("reason") or "").strip(),
        },
        "delta": _quality_delta(m4c, m3c),   # m4 相对 m3：上升=M4 把 context 精炼得更好
        "scored": (m3c is not None or m4c is not None),
    }
    return out


def run_m5(req,
           draft: str,
           m3_output: str,
           final_answer: str,
           deleted_blocks: list[dict],
           engineer_answer: Optional[str],
           m3_feedback: list[dict] | None = None,
           baseline_answer: Optional[str] = None,   # 不使用，保留签名兼容
           baseline_model: Optional[str] = None,
           judge_model_cfg=None,
           judge_temperature: float = 0.1,
           raw_llm_fn=None,
           strip_fences_fn=None,
           enable_engineer_eval: bool = False,        # ★ 新增：工程师对照总开关（默认关）
           compressed_blocks: Optional[list[dict]] = None,  # ★ 新增：M4 压缩块 before/after（预留）
           ) -> dict:
    """执行 M5 评估（v3：开关化工程师对照 + M4 长度/简洁分）。

    enable_engineer_eval=False（默认）:
        跳过一致率 / 纯度 / 覆盖率 / M4删除精度（均需工程师答案）；
        仍计算 M3冗余度、M3核心质量、M4长度、M4简洁分。
    enable_engineer_eval=True 但无工程师答案:
        自动降级为 False 并记 log。
    """
    if raw_llm_fn is None or strip_fences_fn is None:
        from pipeline import _raw_llm as raw_llm_fn  # type: ignore
        from pipeline import _strip_fences as strip_fences_fn  # type: ignore

    category = req.category.primary.value
    is_generation = (category == TaskCategory.GENERATION.value)

    # —— 工程师对照开关的实际生效判定 ——
    have_engineer = bool(engineer_answer and str(engineer_answer).strip())
    do_engineer = enable_engineer_eval and have_engineer
    if enable_engineer_eval and not have_engineer:
        logger.info("M5: enable_engineer_eval=True 但无工程师答案，自动降级为仅客观/绝对指标。")

    # 抽取 Final 核心：优先用 final_answer 的 ⟦CORE⟧，回退到 m3_output 的核心，再回退整段
    final_core = (extract_core_by_markers(final_answer)
                  or extract_core_by_markers(m3_output)
                  or _strip_core_markers(final_answer)
                  or _strip_core_markers(m3_output))

    m2_full = (draft or "").strip()
    m2_core_hint = None  # M2 无标记，置 None 让 judge 自行抽取

    m3_pruned = m3_pruned_core(m3_feedback)

    judge_calls = 0
    raw: Optional[dict] = None

    # ===== A. 工程师对照 judge（仅开关开启且有工程师答案时） =====
    if do_engineer:
        prompt = _build_m5_prompt(
            req             = req,
            engineer_answer = engineer_answer.strip()[:6000],
            final_core      = (final_core or "")[:6000],
            m2_full         = m2_full[:6000],
            m2_core_hint    = m2_core_hint,
            deleted_blocks  = deleted_blocks,
            category        = category,
            is_generation   = is_generation,
        )
        for attempt, temp in enumerate((judge_temperature, 0.0)):
            try:
                judge_calls += 1
                resp = raw_llm_fn(prompt, system=_M5_SYSTEM,
                                  temperature=temp, model_cfg=judge_model_cfg)
                raw = json.loads(strip_fences_fn(resp))
                if isinstance(raw, dict):
                    break
            except Exception as exc:
                logger.warning("M5 judge attempt %d failed: %s", attempt + 1, exc)
                raw = None
        if raw is None:
            # 工程师对照解析失败：不致命，降级为仅客观/绝对指标继续
            logger.warning("M5: 工程师对照 judge 解析失败，降级为仅客观/绝对指标。")
            do_engineer = False

    # ===== B. M3/M2 核心绝对质量（始终） =====
    quality = _run_quality_judge(
        req               = req,
        m3_core           = (final_core or ""),
        m2_full           = m2_full,
        category          = category,
        judge_model_cfg   = judge_model_cfg,
        judge_temperature = judge_temperature,
        raw_llm_fn        = raw_llm_fn,
        strip_fences_fn   = strip_fences_fn,
    )
    if quality is not None:
        judge_calls += 1

    # ===== C. M4 context 效率分（始终；两侧 context 皆空时跳过） =====
    #   M3 context = m3_output 去掉 ⟦CORE⟧ 段（M4 处理前的 context）
    #   M4 context = final_answer 去掉 ⟦CORE⟧ 段（M4 精炼后的 context）
    m3_context = _noncore_text(m3_output or "")
    m4_context = _noncore_text(final_answer or "")
    m4_concision = _run_m4_ctxqual_judge(
        req               = req,
        category          = category,
        m3_context        = m3_context,
        m4_context        = m4_context,
        judge_model_cfg   = judge_model_cfg,
        judge_temperature = judge_temperature,
        raw_llm_fn        = raw_llm_fn,
        strip_fences_fn   = strip_fences_fn,
    )
    if m4_concision is not None:
        judge_calls += 1

    # ===== D. M4 长度（客观，不调 LLM） =====
    m4_length = _compute_m4_length(m3_output, deleted_blocks, compressed_blocks)

    result = _postprocess_m5(
        raw, deleted_blocks, m3_pruned,
        quality=quality,
        m4_concision=m4_concision,
        m4_length=m4_length,
        engineer_eval=do_engineer,
    )
    result["core_pruning"] = _core_pruning_stats(m3_feedback)
    result["judge_calls"] = judge_calls
    result["engineer_eval_enabled"] = do_engineer
    if judge_model_cfg is not None:
        result["judge_model"] = getattr(judge_model_cfg, "model", "unknown")
    return result


# =============================================================================
# 六、M5 指标聚合（分类别 × 分层）
# =============================================================================

class M5Aggregator:
    """线程安全的 M5 指标聚合，按【类别】×【M3是否剔除核心】分层。"""

    def __init__(self, ablation_mode: str = "full"):
        self._lock = threading.Lock()
        self._ablation_mode = ablation_mode
        self._data: dict[str, dict[str, dict]] = {"pruned": {}, "untouched": {}}

    @staticmethod
    def _blank() -> dict:
        return {
            "records": 0,
            # M3 核心淘汰统计（不依赖工程师答案）
            "prune_split_sum": 0, "prune_kept_sum": 0, "prune_elim_sum": 0,
            "prune_n": 0, "prune_ratio_sum": 0.0, "prune_ratio_n": 0,
            "m3_consistent": 0, "m3_partial": 0, "m3_contradictory": 0, "m3_na": 0,
            "m2_consistent": 0, "m2_partial": 0, "m2_contradictory": 0, "m2_na": 0,
            "improved": 0, "unchanged": 0, "regressed": 0,
            "deleted_total": 0, "deleted_tn": 0, "deleted_fn": 0,
            "precision_sum": 0.0, "precision_n": 0,
            "purity_m2_sum": 0.0, "purity_m3_sum": 0.0, "purity_n": 0,
            "purity_improved": 0, "purity_unchanged": 0, "purity_regressed": 0,
            "approaches_m2_sum": 0, "approaches_m3_sum": 0, "approaches_n": 0,
            "recall_m2_sum": 0.0, "recall_m3_sum": 0.0, "recall_n": 0,
            "points_recovered": 0, "points_lost": 0,
            "quality_m2_sum": 0, "quality_m3_sum": 0, "quality_n": 0,
            "quality_improved": 0, "quality_unchanged": 0, "quality_regressed": 0,
            # ★ 新增：M4 长度（客观）
            "m4_noncore_before_sum": 0, "m4_removed_sum": 0,
            "m4_redratio_sum": 0.0, "m4_redratio_n": 0,
            # ★ M4 context 效率分（绝对：信息富度+简洁，合成一分）：M3ctx、M4ctx 各累加 + 升降计数
            "m4ctx_m3_sum": 0, "m4ctx_m4_sum": 0, "m4ctx_n": 0,
            "m4ctx_improved": 0, "m4ctx_unchanged": 0, "m4ctx_regressed": 0,
        }

    @staticmethod
    def _bucket(verdict: str) -> str:
        v = (verdict or "").strip().lower()
        if v in ("consistent", "functionally_equivalent"):
            return "consistent"
        if v == "partially_consistent":
            return "partial"
        if v == "contradictory":
            return "contradictory"
        return "na"

    def add(self, category: str, m5: dict) -> None:
        if not m5 or m5.get("m5_skipped") or m5.get("m5_error"):
            return
        cons = m5.get("consistency", {}) or {}
        engineer_eval = bool(cons.get("engineer_eval", True)) and ("engineer_eval" not in cons or cons.get("engineer_eval") is not False)
        # 更稳妥：以结果里的标志为准
        engineer_eval = bool(m5.get("engineer_eval_enabled", False)) or ("m3_core_vs_engineer" in cons)
        layer = "pruned" if cons.get("m3_pruned_core") else "untouched"
        with self._lock:
            d = self._data[layer].setdefault(category, self._blank())
            d["records"] += 1

            # ---- M3 核心淘汰统计：不依赖工程师答案，始终尝试累加 ----
            cp = m5.get("core_pruning", {}) or {}
            if cp.get("split_candidates") is not None:
                d["prune_split_sum"] += cp.get("split_candidates", 0) or 0
                d["prune_kept_sum"]  += cp.get("kept_candidates", 0) or 0
                d["prune_elim_sum"]  += cp.get("eliminated_candidates", 0) or 0
                d["prune_n"]         += 1
            if cp.get("elimination_ratio") is not None:
                d["prune_ratio_sum"] += cp["elimination_ratio"]
                d["prune_ratio_n"]   += 1

            # ---- 依赖工程师答案的统计：仅在该样本启用了工程师对照时累加 ----
            if engineer_eval and "m3_core_vs_engineer" in cons:
                m3b = self._bucket(cons.get("m3_core_vs_engineer", {}).get("verdict", ""))
                m2b = self._bucket(cons.get("m2_core_vs_engineer", {}).get("verdict", ""))
                d[f"m3_{m3b}"] += 1
                d[f"m2_{m2b}"] += 1

                delta = cons.get("delta", "unchanged")
                if delta in ("improved", "unchanged", "regressed"):
                    d[delta] += 1

                prec = cons.get("precision", {}) or {}
                if (prec.get("m2_core") is not None and prec.get("m3_core") is not None
                        and not prec.get("count_unreliable")):
                    d["purity_m2_sum"] += prec["m2_core"]
                    d["purity_m3_sum"] += prec["m3_core"]
                    d["purity_n"]      += 1
                    pd = prec.get("delta", "unchanged")
                    if pd in ("improved", "unchanged", "regressed"):
                        d[f"purity_{pd}"] += 1

                m3blk = cons.get("m3_core_vs_engineer", {})
                m2blk = cons.get("m2_core_vs_engineer", {})
                if not prec.get("count_unreliable"):
                    try:
                        a2 = int(m2blk.get("total_approaches"))
                        a3 = int(m3blk.get("total_approaches"))
                        if a2 >= 1 and a3 >= 1:
                            d["approaches_m2_sum"] += a2
                            d["approaches_m3_sum"] += a3
                            d["approaches_n"]      += 1
                    except (TypeError, ValueError):
                        pass

                dela = (m5.get("deletion_analysis", {}) or {}).get("summary", {})
                d["deleted_total"] += dela.get("total_deleted", 0)
                d["deleted_tn"]    += dela.get("true_negative", 0)
                d["deleted_fn"]    += dela.get("false_negative", 0)
                if dela.get("precision") is not None:
                    d["precision_sum"] += dela["precision"]
                    d["precision_n"]   += 1

                cov = (m5.get("coverage", {}) or {}).get("summary", {})
                if cov.get("recall_m3_core") is not None:
                    d["recall_m2_sum"] += cov.get("recall_m2_core") or 0.0
                    d["recall_m3_sum"] += cov.get("recall_m3_core") or 0.0
                    d["recall_n"]      += 1
                d["points_recovered"] += cov.get("points_recovered_by_m3", 0)
                d["points_lost"]      += cov.get("points_lost_by_m3", 0)

            # ---- 不依赖工程师答案的统计：始终累加 ----
            # 冗余度（核心方案数 M2 vs M3）—— 仅在有工程师对照、计数可信时才有方案数；
            # 关闭工程师对照时无方案计数来源，approaches_n 不增（冗余度列将显示 —）。

            cq = m5.get("core_quality", {}) or {}
            m2q = (cq.get("m2_core") or {}).get("score")
            m3q = (cq.get("m3_core") or {}).get("score")
            if m2q is not None and m3q is not None:
                d["quality_m2_sum"] += m2q
                d["quality_m3_sum"] += m3q
                d["quality_n"]      += 1
                qd = cq.get("delta", "unchanged")
                if qd in ("improved", "unchanged", "regressed"):
                    d[f"quality_{qd}"] += 1

            ml = m5.get("m4_length", {}) or {}
            d["m4_noncore_before_sum"] += ml.get("noncore_chars_before", 0) or 0
            d["m4_removed_sum"]        += ml.get("removed_chars", 0) or 0
            if ml.get("reduction_ratio") is not None:
                d["m4_redratio_sum"] += ml["reduction_ratio"]
                d["m4_redratio_n"]   += 1

            mc = m5.get("m4_context_quality", {}) or {}
            m3c = (mc.get("m3_context") or {}).get("score")
            m4c = (mc.get("m4_context") or {}).get("score")
            if m3c is not None and m4c is not None:
                d["m4ctx_m3_sum"] += m3c
                d["m4ctx_m4_sum"] += m4c
                d["m4ctx_n"]      += 1
                md = mc.get("delta", "unchanged")
                if md in ("improved", "unchanged", "regressed"):
                    d[f"m4ctx_{md}"] += 1

    # ----- 渲染 -----
    @staticmethod
    def _consist_rate(d, prefix):
        ok  = d[f"{prefix}_consistent"]
        tot = d[f"{prefix}_consistent"] + d[f"{prefix}_partial"] + d[f"{prefix}_contradictory"]
        return f"{ok * 100 / tot:.1f}%" if tot else "—"

    def _render_layer(self, layer: str, layer_title: str, color, pad_fn) -> None:
        with self._lock:
            cats = {k: dict(v) for k, v in self._data[layer].items()}
        cols = [("类别", 14), ("样本", 6), ("M3核心一致率", 14), ("M2核心一致率", 14),
                ("改善率", 9), ("劣化率", 9), ("M4删除精度", 12),
                ("纯度M2→M3", 14), ("要点召回M2→M3", 16), ("补回/丢失", 12)]
        line_w = sum(w for _, w in cols)

        print("\n" + "=" * line_w)
        print(color.bold(f"  M5 一致性（{layer_title}）"))
        print("=" * line_w)
        if not cats:
            print(color.dim("  （本层无样本）"))
            print("=" * line_w)
            return
        print("".join(pad_fn(h, w) for h, w in cols))
        print("-" * line_w)

        def row(name, d, bold=False):
            recs = d["records"]
            imp = f"{d['improved'] * 100 / recs:.0f}%" if recs else "—"
            reg = f"{d['regressed'] * 100 / recs:.0f}%" if recs else "—"
            prec = (f"{d['precision_sum'] * 100 / d['precision_n']:.1f}%"
                    if d["precision_n"] else "—")
            if d["purity_n"]:
                purity_str = (f"{d['purity_m2_sum'] * 100 / d['purity_n']:.0f}%"
                              f"→{d['purity_m3_sum'] * 100 / d['purity_n']:.0f}%")
            else:
                purity_str = "—"
            if d["recall_n"]:
                rec_str = (f"{d['recall_m2_sum'] * 100 / d['recall_n']:.0f}%"
                           f"→{d['recall_m3_sum'] * 100 / d['recall_n']:.0f}%")
            else:
                rec_str = "—"
            recov = f"{d['points_recovered']}/{d['points_lost']}"
            cells = [name, recs, self._consist_rate(d, "m3"), self._consist_rate(d, "m2"),
                     imp, reg, prec, purity_str, rec_str, recov]
            line = "".join(pad_fn(c, w) for c, (_, w) in zip(cells, cols))
            return color.bold(line) if bold else line

        total = self._blank()
        for cat in sorted(cats, key=lambda c: cats[c]["records"], reverse=True):
            d = cats[cat]
            for k in total:
                total[k] += d[k]
            print(row(cat, d))
        print("-" * line_w)
        print(row("总计", total, bold=True))
        print("=" * line_w)

    def _merge_layers_by_cat(self) -> dict:
        """合并分层①②，按类别汇总（用于简洁性=全样本维度）。"""
        with self._lock:
            merged = {}
            for layer in ("pruned", "untouched"):
                for cat, d in self._data[layer].items():
                    if cat not in merged:
                        merged[cat] = self._blank()
                    for k in merged[cat]:
                        merged[cat][k] += d[k]
            return merged

    def _render_three_dim(self, color, pad_fn) -> None:
        """三维度主表：冗余度↓ / 一致性↑(一致率+纯度+核心质量) / 简洁性↑(M4长度+简洁分)。
        纵轴=类别；冗余度与一致性按『分层①』口径，简洁性为全样本。
        例外：no_debate 模式下核心质量取全样本(因该模式无 pruning,样本全在 untouched 层)。"""
        with self._lock:
            pruned = {k: dict(v) for k, v in self._data["pruned"].items()}
        merged = self._merge_layers_by_cat()

        cols = [("类别", 14),
                ("①样本", 7), ("M3候选保留", 14), ("M3淘汰率", 10),
                ("冗余度方案M2→M3", 18),
                ("一致率M2→M3", 16), ("纯度M2→M3", 14), ("核心质量M2→M3", 16),
                ("全样本", 8), ("M4长度前→后", 15), ("M4缩减率", 11),
                ("M4context质量M3→M4", 20), ("M4删除精度", 12)]
        line_w = sum(w for _, w in cols)

        print("\n" + "=" * line_w)
        print(color.bold("  M5 方法有效性（三维度）：冗余度↓  一致性↑(一致率+纯度+核心质量)  简洁性↑(M4长度+context质量)"))
        print("=" * line_w)
        print("".join(pad_fn(h, w) for h, w in cols))
        print("-" * line_w)

        def pruning_flow(d):
            if d["prune_n"]:
                return (f"{d['prune_split_sum'] / d['prune_n']:.2f}"
                        f"→{d['prune_kept_sum'] / d['prune_n']:.2f}")
            return "—"

        def pruning_ratio(d):
            return (f"{d['prune_ratio_sum'] * 100 / d['prune_ratio_n']:.0f}%"
                    if d["prune_ratio_n"] else "—")

        def redundancy(d):
            if d["approaches_n"]:
                return (f"{d['approaches_m2_sum'] / d['approaches_n']:.2f}"
                        f"→{d['approaches_m3_sum'] / d['approaches_n']:.2f}")
            return "—"

        def purity(d):
            if d["purity_n"]:
                return (f"{d['purity_m2_sum'] * 100 / d['purity_n']:.0f}%"
                        f"→{d['purity_m3_sum'] * 100 / d['purity_n']:.0f}%")
            return "—"

        def consist(d):
            if (d["m2_consistent"] + d["m2_partial"] + d["m2_contradictory"]) == 0:
                return "—"
            return f"{self._consist_rate(d, 'm2')}→{self._consist_rate(d, 'm3')}"

        def quality(d):
            if d["quality_n"]:
                return (f"{d['quality_m2_sum'] / d['quality_n']:.2f}"
                        f"→{d['quality_m3_sum'] / d['quality_n']:.2f}")
            return "—"

        def m4_len_flow(d):
            """平均非核心 context 字符数：M4 处理前 → M4 处理后。"""
            if d["records"]:
                before = d["m4_noncore_before_sum"] / d["records"]
                after = max(0.0, (d["m4_noncore_before_sum"] - d["m4_removed_sum"]) / d["records"])
                return f"{before:.0f}→{after:.0f}"
            return "—"

        def m4_red(d):
            """总字符口径缩减率：总去除字符 / M3 非核心总字符。"""
            before = d["m4_noncore_before_sum"]
            return (f"{d['m4_removed_sum'] * 100 / before:.0f}%" if before else "—")

        def m4_ctxqual(d):
            if d["m4ctx_n"]:
                return (f"{d['m4ctx_m3_sum'] / d['m4ctx_n']:.2f}"
                        f"→{d['m4ctx_m4_sum'] / d['m4ctx_n']:.2f}")
            return "—"

        def m4_prec(d):
            return (f"{d['precision_sum'] * 100 / d['precision_n']:.1f}%"
                    if d["precision_n"] else "—")

        def row(name, dp, dm, bold=False):
            # 核心质量：no_debate 模式无 pruning,样本全在 untouched 层,故取全样本(dm)；
            # 其他模式仍取分层①(dp),与冗余度/一致率/纯度口径一致。
            quality_src = dm if self._ablation_mode == "no_debate" else dp
            cells = [name,
                     dp["records"], pruning_flow(dp), pruning_ratio(dp), redundancy(dp),
                     consist(dp), purity(dp), quality(quality_src),
                     dm["records"], m4_len_flow(dm), m4_red(dm), m4_ctxqual(dm), m4_prec(dm)]
            line = "".join(pad_fn(c, w) for c, (_, w) in zip(cells, cols))
            return color.bold(line) if bold else line

        all_cats = sorted(merged, key=lambda c: merged[c]["records"], reverse=True)
        tot_p, tot_m = self._blank(), self._blank()
        for cat in all_cats:
            dp = pruned.get(cat, self._blank())
            dm = merged[cat]
            for k in tot_p: tot_p[k] += dp[k]
            for k in tot_m: tot_m[k] += dm[k]
            print(row(cat, dp, dm))
        print("-" * line_w)
        print(row("总计", tot_p, tot_m, bold=True))
        print("=" * line_w)
        print(color.dim("  ① M3候选保留/淘汰率 = M3最新一轮核心候选数→保留数、淘汰比例（不依赖工程师答案）；"))
        print(color.dim("  ② 冗余度 = M3核心相对M2核心 的平均方案数 M2→M3（仅分层①；需开启工程师对照才有方案计数）；"))
        print(color.dim("  ③ 一致性 = 一致率 + 纯度 + 核心质量（仅分层①）："))
        print(color.dim("       一致率 = 与工程师核心一致的样本占比 M2→M3（需开启工程师对照）；"))
        print(color.dim("       纯度   = 一致方案/总方案 M2→M3（需开启工程师对照）；"))
        print(color.dim("       核心质量 = M3核心绝对质量评分(1-5,不看工程师答案) M2→M3；"
                        "full/no_compress/no_taskaware 取分层①,no_debate 取全样本(因其无 pruning)；"))
        print(color.dim(
            "  ④ 简洁性（M4）= M4长度前→后 + M4缩减率(客观) + M4context质量(绝对1-5,不看工程师) + 删除精度（删除精度需工程师对照）；"))
        print(color.dim("       M4长度前→后 = 平均非核心 context 字符数：M3输出中的非核心长度 → M4处理后的非核心长度；"))
        print(color.dim("       M4缩减率 = 总去除字符 / M3非核心总字符（删除+压缩，和 M4长度前→后 同一字符口径）；"))
        print(color.dim(
            "       M4context质量 = context 的合成质量分(必要性/有用性 + 简洁度 + 非冗余) M3context→M4context（上升=M4 把 context 精炼得更好）；"))
        print(color.dim("  口径：冗余度与一致性聚焦『分层①』；简洁性为全样本。带『需开启工程师对照』的列在关闭时显示 —。"))

    def render(self, color, pad_fn) -> None:
        has_any = any(self._data[l] for l in ("pruned", "untouched"))
        if not has_any:
            print("\n（无 M5 评估统计）\n")
            return

        # ★ 三维度主表（论文用）
        self._render_three_dim(color, pad_fn)

        # 主表 / 副表（一致性细分，仅在有工程师对照样本时有内容）
        self._render_layer("pruned", "分层①：M3 实际剔除过核心的样本", color, pad_fn)
        self._render_layer("untouched", "分层②：M3 未改动核心的样本（M2≈M3，仅作对照）", color, pad_fn)

        print(color.dim("说明：一致率/纯度/要点召回/删除精度 仅在开启工程师对照(enable_engineer_eval=True)时产生；"))
        print(color.dim("      核心质量、M4缩减率、M4context质量 不依赖工程师答案，始终产生；"))
        print(color.dim("      M3核心/M2核心 = M3剔除后核心 / M2原始核心；"))
        print(color.dim("      ★ 论点证据看『分层①』：若改善率 > 劣化率，说明 M3 剔除错误核心确实提升了与工程师的一致性。"))
        print()
        print(color.dim("  口径：冗余度与一致性聚焦『分层①』；简洁性为全样本。带『需开启工程师对照』的列在关闭时显示 —。"))