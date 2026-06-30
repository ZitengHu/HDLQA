

from __future__ import annotations

import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional

import httpx
from openai import OpenAI

try:
    import autogen  # noqa: F401
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
except ImportError:
    print("Please install autogen first: pip install pyautogen")
    sys.exit(1)

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """单个模型端点配置。"""
    api_key:  str = ""
    base_url: str = ""
    model:    str = "deepseek-v3"


@dataclass
class ModelRouteConfig:
    """模型路由配置 —— 每个阶段都有明确的模型入口。

    回退规则(__post_init__ 实现),不填(None)时:
      readability/consistency/concise/codegen -> default     (M2 能力路由)
      m3                                       -> default     (M3 验证)
      m4                                       -> concise -> default (M4 筛检)
      judge                                    -> default     (M5 评估)
    只配 default 即可跑通;需给某阶段单独换模型时再显式指定对应字段。
    """
    # ---- 通用兜底 ----
    default:     ModelConfig = None      # M1 分类 + 所有未指定字段的兜底

    # ---- M2 草稿生成:按任务类别路由 ----
    readability: ModelConfig = None      # M2 conceptual
    consistency: ModelConfig = None      # M2 debugging
    concise:     ModelConfig = None      # M2 optimization
    codegen:     ModelConfig = None      # M2 generation

    # ---- M3 分级验证:拆分 / 辩论 / 排序 / 组装(全部走此字段)----
    m3:          ModelConfig = None      # M3 专属入口
    # ---- M3 分级验证(可整体配 m3,或按子环节细分覆盖)----
    m3: ModelConfig = None  # M3 整层默认(四个子环节的兜底)
    m3_split: ModelConfig = None  # 核心/非核心拆分        — 中等
    m3_debate: ModelConfig = None  # Proposer/Challenger/Reflector 三方辩论 — 最重
    m3_rank: ModelConfig = None  # Top-N 排序选优         — 中等
    m3_assemble: ModelConfig = None  # 多候选组装            — 最轻

    # ---- M4 精炼润色:内容分解 + 非核心筛检 ----
    m4:          ModelConfig = None      # M4 专属入口

    # ---- M5 对照评估 ----
    judge:       ModelConfig = None      # M5 LLM-as-Judge

    def __post_init__(self):
        if self.default is None:
            self.default = ModelConfig()
        # 第一档:M2 路由 + M3 基准 + M5 -> default
        for field in ("readability", "consistency", "concise",
                      "codegen", "m3", "judge"):
            if getattr(self, field) is None:
                object.__setattr__(self, field, self.default)
        # 第二档:M3 子环节 -> m3(此时 m3 已非 None,可一路回退到 default)
        for field in ("m3_split", "m3_debate", "m3_rank", "m3_assemble"):
            if getattr(self, field) is None:
                object.__setattr__(self, field, self.m3)
        # M4 -> concise -> default
        if self.m4 is None:
            object.__setattr__(self, "m4", self.concise)


@dataclass
class PipelineConfig:
    """流水线全局配置。"""
    input_path:  str = ""
    output_path: str = ""
    models: ModelRouteConfig = None
    max_verify_rounds: int   = 2
    max_autogen_turns: int   = 12
    llm_temperature:   float = 0.0
    batch_max_items: int   = 0
    batch_delay_sec: float = 1.0
    concurrency:     int   = 4
    # ---- 保存参数 ----------------------------------------------------------
    auto_save_interval: int = 5   # 每处理 N 条自动保存一次中间结果

    # ---- M5 评估参数 -------------------------------------------------------
    enable_m5:                bool  = True
    m5_accepted_answer_field: str   = "accepted_answer"
    m5_judge_temperature:     float = 0.0
    m5_baseline_model:        str   = ""   # 空=自动选BLEU最高的LLM作baseline
    m5_enable_engineer_eval:  bool  = False  # ★ 新增:M5 工程师对照总开关(默认关)

    ablation_mode: str = "full"

    def __post_init__(self):
        if self.models is None:
            self.models = ModelRouteConfig()


_cfg: PipelineConfig = PipelineConfig()


def _apply_config(cfg: PipelineConfig) -> None:
    global _cfg
    _cfg = cfg


def _llm_config_for(model_cfg: ModelConfig,
                    temperature: Optional[float] = None) -> dict:
    return {
        "config_list": [
            {
                "model":    model_cfg.model,
                "api_key":  model_cfg.api_key,
                "base_url": model_cfg.base_url,
            }
        ],
        "temperature": temperature if temperature is not None else _cfg.llm_temperature,
        "timeout": 120,
    }


def _default_llm_config() -> dict:
    return _llm_config_for(_cfg.models.default)


def _raw_llm(prompt: str,
             system: str = "You are a helpful assistant.",
             temperature: Optional[float] = None,
             model_cfg: Optional[ModelConfig] = None) -> str:
    mc = model_cfg or _cfg.models.default
    t  = temperature if temperature is not None else _cfg.llm_temperature
    client = OpenAI(
        base_url=mc.base_url,
        api_key=mc.api_key,
        http_client=httpx.Client(timeout=90.0),
    )
    resp = client.chat.completions.create(
        model=mc.model,
        temperature=t,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


class _C:
    _enabled = sys.stdout.isatty()

    @staticmethod
    def _w(code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if _C._enabled else text

    @staticmethod
    def bold(t):   return _C._w("1", t)
    @staticmethod
    def dim(t):    return _C._w("2", t)
    @staticmethod
    def cyan(t):   return _C._w("36", t)
    @staticmethod
    def green(t):  return _C._w("32", t)
    @staticmethod
    def red(t):    return _C._w("31", t)
    @staticmethod
    def yellow(t): return _C._w("33", t)
    @staticmethod
    def blue(t):   return _C._w("34", t)


_STAGE_META = {
    "M1": ("M1", "需求分类", "📋"),
    "M2": ("M2", "草稿生成", "✏️"),
    "M3": ("M3", "分级验证", "🔍"),
    "M4": ("M4", "精炼润色", "✨"),
    "M5": ("M5", "对照评估", "⚖️"),
}

_VERBOSE: bool = True
_PRINT_LOCK = threading.Lock()


def _set_verbose(v: bool) -> None:
    global _VERBOSE
    _VERBOSE = v


def _safe_print(line: str = "") -> None:
    with _PRINT_LOCK:
        print(line)


def _pad(s, width: int) -> str:
    s = str(s)
    return s + " " * max(0, width - _display_len(s))


def _display_len(s: str) -> int:
    n = 0
    for ch in s:
        n += 2 if ord(ch) > 0x2E80 else 1
    return n


def _stage_banner(stage: str, extra: str = "") -> None:
    if not _VERBOSE:
        return
    code, name, icon = _STAGE_META.get(stage, (stage, "", "•"))
    width = 65
    label = f" {icon}  阶段 {code} · {name}"
    if extra:
        label += f"  —  {extra}"
    pad = width - 2 - _display_len(label)
    print()
    print(_C.cyan("╔" + "═" * (width - 2) + "╗"))
    print(_C.cyan("║") + _C.bold(label) + " " * max(pad, 0) + _C.cyan("║"))
    print(_C.cyan("╚" + "═" * (width - 2) + "╝"))


def _substep(msg: str) -> None:
    if not _VERBOSE:
        return
    print(_C.dim("   ├─ ") + msg)


def _result_line(ok: bool, msg: str) -> None:
    if not _VERBOSE:
        return
    mark = _C.green("✔") if ok else _C.red("✗")
    print(f"   {mark} {msg}")


def _detail(msg: str) -> None:
    if not _VERBOSE:
        return
    print(_C.dim(f"        {msg}"))


def _block(title: str, body: str) -> None:
    if not _VERBOSE:
        return
    width = 65
    print(_C.dim("   ┌─ ") + _C.bold(title) + " " + _C.dim("─" * max(width - 6 - _display_len(title), 0)))
    for line in (body or "").splitlines() or [""]:
        print(_C.dim("   │ ") + line)
    print(_C.dim("   └" + "─" * (width - 4)))


class ProgressTracker:
    """线程安全的进度跟踪器，带 ETA 估算和实时进度条。"""

    def __init__(self, total: int):
        self._lock = threading.Lock()
        self._total = total
        self._done = 0
        self._ok = 0
        self._fail = 0
        self._skip = 0
        self._start = time.time()

    def update(self, ok: bool = True, skip: bool = False) -> None:
        with self._lock:
            self._done += 1
            if skip:
                self._skip += 1
            elif ok:
                self._ok += 1
            else:
                self._fail += 1

    def render(self) -> str:
        with self._lock:
            done, total = self._done, self._total
            ok, fail, skip = self._ok, self._fail, self._skip

        elapsed = time.time() - self._start
        pct = done / total * 100 if total else 0
        bar_len = 30
        filled = int(bar_len * done / total) if total else 0
        bar = "█" * filled + "░" * (bar_len - filled)

        if done > 0 and done < total:
            eta_sec = elapsed / done * (total - done)
            if eta_sec >= 3600:
                eta_str = f"{eta_sec / 3600:.1f}h"
            elif eta_sec >= 60:
                eta_str = f"{eta_sec / 60:.1f}m"
            else:
                eta_str = f"{eta_sec:.0f}s"
        elif done >= total:
            eta_str = "done"
        else:
            eta_str = "—"

        elapsed_str = f"{elapsed:.0f}s"

        return (
            f"\r  {bar} {pct:5.1f}%  "
            f"[{done}/{total}]  "
            f"✔{ok} ✗{fail} ⊘{skip}  "
            f"{elapsed_str} elapsed  ETA {eta_str}  "
        )

    def print_bar(self) -> None:
        with _PRINT_LOCK:
            sys.stdout.write(self.render())
            sys.stdout.flush()

    def finish(self) -> None:
        with _PRINT_LOCK:
            sys.stdout.write(self.render() + "\n")
            sys.stdout.flush()


class TaskCategory(str, Enum):
    CONCEPTUAL   = "conceptual"
    DEBUGGING    = "debugging"
    GENERATION   = "generation"
    OPTIMIZATION = "optimization"


_FALLBACK_CATEGORY = TaskCategory.DEBUGGING


class VerificationTag(str, Enum):
    SEMANTIC         = "semantic_correctness"
    SYNTAX           = "syntax_check"
    SYNTHESIZABILITY = "synthesizability_check"
    FUNC_ALIGN       = "functional_alignment"
    OPT_VALIDITY     = "optimization_validity"


class RiskLevel(str, Enum):
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


@dataclass
class RiskProfile:
    readability_risk: RiskLevel = RiskLevel.MEDIUM
    consistency_risk: RiskLevel = RiskLevel.MEDIUM
    redundancy_risk:  RiskLevel = RiskLevel.MEDIUM
    conciseness_risk: RiskLevel = RiskLevel.MEDIUM

    def to_dict(self) -> dict:
        return {
            "readability_risk": self.readability_risk.value,
            "consistency_risk": self.consistency_risk.value,
            "redundancy_risk":  self.redundancy_risk.value,
            "conciseness_risk": self.conciseness_risk.value,
        }

    @property
    def high_risks(self) -> list[str]:
        return [name for name in
                ("readability_risk", "consistency_risk", "redundancy_risk", "conciseness_risk")
                if getattr(self, name) == RiskLevel.HIGH]


@dataclass
class AnswerPolicy:
    target_length:         str  = "medium"
    include_code:          str  = "only_if_needed"
    include_testbench:     bool = False
    list_multiple_options: bool = False

    def to_dict(self) -> dict:
        return {
            "target_length":                    self.target_length,
            "whether_to_include_code":          self.include_code,
            "whether_to_include_testbench":     "yes" if self.include_testbench else "no",
            "whether_to_list_multiple_options": "yes" if self.list_multiple_options else "no",
        }


@dataclass
class CategoryResult:
    primary:   TaskCategory
    secondary: Optional[TaskCategory] = None


@dataclass
class TaskRequirements:
    raw_question:          str
    title:                 str
    category:              CategoryResult
    user_intent:           str
    required_output:       list[str]
    verification:          list[VerificationTag]
    risk_profile:          RiskProfile
    answer_policy:         AnswerPolicy
    formatted_description: str
    dataset_label:         dict
    classification_review: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "category": {
                "primary":   self.category.primary.value,
                "secondary": self.category.secondary.value if self.category.secondary else None,
            },
            "user_intent":           self.user_intent,
            "required_output":       self.required_output,
            "verification":          [v.value for v in self.verification],
            "risk_profile":          self.risk_profile.to_dict(),
            "answer_policy":         self.answer_policy.to_dict(),
            "formatted_description": self.formatted_description,
            "title":                 self.title,
            "raw_question":          self.raw_question,
            "dataset_label":         self.dataset_label,
            "classification_review": self.classification_review or {},
        }


_CATEGORY_PRIORS: dict[str, dict] = {
    TaskCategory.CONCEPTUAL.value: {
        "risk":   dict(readability="medium", consistency="medium",
                       redundancy="medium", conciseness="high"),
        "policy": dict(target_length="short", include_code="only_if_needed",
                       include_testbench=False, list_multiple_options=False),
        "verification": [VerificationTag.SEMANTIC],
    },
    TaskCategory.DEBUGGING.value: {
        "risk":   dict(readability="medium", consistency="high",
                       redundancy="high", conciseness="medium"),
        "policy": dict(target_length="medium", include_code="yes",
                       include_testbench=False, list_multiple_options=False),
        "verification": [VerificationTag.SEMANTIC, VerificationTag.SYNTAX,
                         VerificationTag.SYNTHESIZABILITY, VerificationTag.FUNC_ALIGN],
    },
    TaskCategory.GENERATION.value: {
        "risk":   dict(readability="medium", consistency="high",
                       redundancy="medium", conciseness="high"),
        "policy": dict(target_length="medium", include_code="yes",
                       include_testbench=False, list_multiple_options=False),
        "verification": [VerificationTag.SYNTAX, VerificationTag.SYNTHESIZABILITY,
                         VerificationTag.FUNC_ALIGN],
    },
    TaskCategory.OPTIMIZATION.value: {
        "risk":   dict(readability="medium", consistency="high",
                       redundancy="high", conciseness="high"),
        "policy": dict(target_length="medium", include_code="only_if_needed",
                       include_testbench=False, list_multiple_options=False),
        "verification": [VerificationTag.SEMANTIC, VerificationTag.OPT_VALIDITY,
                         VerificationTag.FUNC_ALIGN],
    },
}


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _coerce_category(value) -> TaskCategory:
    if not value:
        return _FALLBACK_CATEGORY
    alias = {
        "concept": TaskCategory.CONCEPTUAL, "conceptual": TaskCategory.CONCEPTUAL,
        "concept_explanation": TaskCategory.CONCEPTUAL, "explanation": TaskCategory.CONCEPTUAL,
        "debug": TaskCategory.DEBUGGING, "debugging": TaskCategory.DEBUGGING,
        "debug_analysis": TaskCategory.DEBUGGING,
        "code_generation": TaskCategory.GENERATION, "code_gen": TaskCategory.GENERATION,
        "generation": TaskCategory.GENERATION, "generate": TaskCategory.GENERATION,
        "optimization": TaskCategory.OPTIMIZATION, "optimisation": TaskCategory.OPTIMIZATION,
        "optimize": TaskCategory.OPTIMIZATION,
        "comparison": TaskCategory.CONCEPTUAL, "comparison_analysis": TaskCategory.CONCEPTUAL,
    }
    return alias.get(str(value).strip().lower(), _FALLBACK_CATEGORY)


def _ci_get(d: dict, keys) -> Optional[str]:
    low = {str(k).lower(): v for k, v in (d or {}).items()}
    for k in keys:
        if low.get(k):
            return str(low[k])
    return None


def _prior_category(dataset_label: dict | None) -> Optional[str]:
    if not dataset_label:
        return None
    v = _ci_get(dataset_label, ("task_type", "category", "primary", "label", "type"))
    if v:
        return v
    inner = dataset_label.get("classification") or dataset_label.get("category_result")
    return _prior_category(inner) if isinstance(inner, dict) else None


def _prior_reason(dataset_label: dict | None) -> Optional[str]:
    if not dataset_label:
        return None
    v = _ci_get(dataset_label, ("reason", "reasoning", "rationale", "justification", "explanation", "why"))
    if v:
        return v
    inner = dataset_label.get("classification") or dataset_label.get("category_result")
    return _prior_reason(inner) if isinstance(inner, dict) else None


def _parse_risk(data: dict | None, prior: dict) -> RiskProfile:
    data = data or {}

    def pick(field_name: str, short: str) -> RiskLevel:
        val = data.get(field_name) or prior.get(short) or "medium"
        try:
            return RiskLevel(str(val).strip().lower())
        except ValueError:
            return RiskLevel.MEDIUM

    return RiskProfile(
        readability_risk=pick("readability_risk", "readability"),
        consistency_risk=pick("consistency_risk", "consistency"),
        redundancy_risk=pick("redundancy_risk",   "redundancy"),
        conciseness_risk=pick("conciseness_risk", "conciseness"),
    )


def _parse_policy(data: dict | None, prior: dict) -> AnswerPolicy:
    data = data or {}

    def yn(field_name: str, default_bool: bool) -> bool:
        v = data.get(field_name)
        if v is None:
            return default_bool
        return str(v).strip().lower() in ("yes", "true", "1")

    target = str(data.get("target_length") or prior.get("target_length", "medium")).strip().lower()
    if target not in ("short", "medium", "detailed"):
        target = "medium"

    code = str(data.get("whether_to_include_code") or prior.get("include_code", "only_if_needed")).strip().lower()
    if code not in ("yes", "no", "only_if_needed"):
        code = "only_if_needed"

    return AnswerPolicy(
        target_length=target,
        include_code=code,
        include_testbench=yn("whether_to_include_testbench", prior.get("include_testbench", False)),
        list_multiple_options=yn("whether_to_list_multiple_options", prior.get("list_multiple_options", False)),
    )


_M1_SYSTEM = (
    "You are a requirements reviewer for Verilog/SystemVerilog/VHDL engineering questions.\n"
    "The dataset ALREADY provides a prior task-type classification together with its reasoning.\n"
    "Your job is to DOUBLE-CHECK that prior classification — NOT to reclassify from scratch:\n"
    "  - If the prior task_type is reasonable for this question, KEEP it.\n"
    "  - Only OVERRIDE it when it is clearly wrong, and then justify the override.\n"
    "Besides the (possibly corrected) task type, also produce the risk profile, answer policy "
    "and verification needs.\n"
    "Output only valid JSON — no markdown fences, no preamble, no extra text."
)

_M1_PROMPT = """Double-check the prior classification of the following HDL engineering question.

<title>{title}</title>
<question>{question}</question>

Prior classification from the dataset (result + reasoning):
{prior_block}

Return ONLY a JSON object with exactly this schema (all fields required):

{{
  "prior_task_type": "<the dataset's prior task_type as you read it, or null if none>",
  "review": "<keep|override>",
  "task_type": "<conceptual|debugging|generation|optimization>",
  "secondary_type": "<same enum, or null>",
  "review_reason": "<one sentence: why the prior is reasonable (keep) OR why it is wrong (override)>",
  "user_intent": "<one sentence: what the user actually wants solved>",
  "required_output": ["<subset of: explanation, code, bug_reason, fix, optimization_suggestion, comparison>"],
  "verification_needs": ["<subset of: semantic_correctness, syntax_check, synthesizability_check, functional_alignment, optimization_validity>"],
  "risk_profile": {{
    "readability_risk": "<low|medium|high>",
    "consistency_risk": "<low|medium|high>",
    "redundancy_risk":  "<low|medium|high>",
    "conciseness_risk": "<low|medium|high>"
  }},
  "answer_policy": {{
    "target_length": "<short|medium|detailed>",
    "whether_to_include_code": "<yes|no|only_if_needed>",
    "whether_to_include_testbench": "<yes|no>",
    "whether_to_list_multiple_options": "<yes|no>"
  }},
  "formatted_description": "<one paragraph: question type, expected output, key constraints>"
}}

Review rules:
- DEFAULT to "keep". Choose "override" ONLY when the prior task_type genuinely mismatches the question.
- When review = "keep", task_type MUST equal the prior task_type.
- When review = "override", task_type is your corrected type and review_reason must explain the mismatch.
- If no prior classification was provided, classify from the question and set review = "keep".

Task-type definitions:
- conceptual   : core ask is what / why / how-it-works (incl. comparing concepts to understand them); usually no new code needed.
- debugging    : existing code/behaviour is given; find the cause of an error/warning/mismatch and fix it.
- generation   : explicitly asks to implement / write an HDL module or snippet.
- optimization : asks to improve existing/working code for PPA, timing, area, power or readability.
Set secondary_type (same enum, or null) only when the question clearly has a meaningful secondary aspect.

Risk-profile guidance (from a study of LLM-vs-engineer HDL answers):
- conceptual   : conciseness_risk is typically HIGH (answers over-explain simple concepts).
- debugging    : redundancy_risk HIGH and consistency_risk HIGH (long summaries, divergence from the real fix).
- generation   : conciseness_risk HIGH and consistency_risk HIGH (extra code/features, guessed APIs, unsolicited testbenches).
- optimization : redundancy_risk HIGH and conciseness_risk HIGH (generic optimisation lectures).
Raise a risk to HIGH when the question makes that failure mode especially likely; lower it when clearly not applicable.

Answer-policy guidance:
- whether_to_include_testbench defaults to "no" unless the user explicitly asks for a testbench/verification.
- whether_to_list_multiple_options defaults to "no"; set "yes" only if the user explicitly wants alternatives compared.
- target_length: conceptual->short, debugging/generation/optimization->medium, "detailed" only when the user clearly wants depth.
- whether_to_include_code: generation/debugging->yes, optimization->only_if_needed, conceptual->only_if_needed or no.

Verification-needs guidance:
- concept/principle reasoning              -> semantic_correctness
- any code is produced or edited           -> syntax_check + synthesizability_check
- a functional behaviour / sim is involved -> functional_alignment
- an optimisation claim is made            -> optimization_validity"""


class M1TaskClassifier:

    _CODE_KW  = re.compile(r"\b(implement|generate|write|design|module|always|assign|wire|reg|instantiat)\b", re.I)
    _DEBUG_KW = re.compile(r"\b(error|bug|warning|fix|wrong|fail|issue|synthesis|latch|removed|unused)\b", re.I)
    _CONC_KW  = re.compile(r"\b(what|why|how|explain|difference|principle|mean|meaning)\b", re.I)
    _OPT_KW   = re.compile(r"\b(optimi[sz]e|improve|faster|reduce|area|timing|power|throughput|ppa|critical\s*path)\b", re.I)

    def classify(self, title: str, question: str,
                 dataset_label: dict | None = None) -> TaskRequirements:
        title         = title.strip()
        question      = question.strip()
        dataset_label = dataset_label or {}
        if not question:
            raise ValueError("Question must not be empty.")
        try:
            prompt = _M1_PROMPT.format(
                title       = title,
                question    = question[:3000],
                prior_block = self._format_prior(dataset_label),
            )
            data = json.loads(_strip_fences(_raw_llm(prompt, system=_M1_SYSTEM, temperature=0.0)))
            return self._build(data, title, question, dataset_label)
        except Exception as exc:
            logger.warning("M1 LLM failed, switching to rule-based fallback: %s", exc)
            return self._fallback(title, question, dataset_label)

    @staticmethod
    def _format_prior(dataset_label: dict) -> str:
        if not dataset_label:
            return "(no prior classification provided — classify from the question itself.)"
        cat    = _prior_category(dataset_label) or "(not specified)"
        reason = _prior_reason(dataset_label)   or "(no reason given)"
        return (f"- prior task_type : {cat}\n"
                f"- prior reasoning : {reason}\n"
                f"- raw label       : {json.dumps(dataset_label, ensure_ascii=False)}")

    def _build(self, data: dict, title: str, question: str,
               dataset_label: dict) -> TaskRequirements:
        prior_cat  = _prior_category(dataset_label)
        review     = (data.get("review") or "").strip().lower()
        model_type = _coerce_category(data.get("task_type"))

        if prior_cat is not None:
            if review == "override":
                primary, review = model_type, "override"
            else:
                primary, review = _coerce_category(prior_cat), "keep"
        else:
            primary, review = model_type, "fresh"

        sec_raw   = data.get("secondary_type")
        secondary = _coerce_category(sec_raw) if sec_raw and sec_raw not in ("null", None) else None
        if secondary == primary:
            secondary = None

        verification: list[VerificationTag] = []
        for v in data.get("verification_needs", []):
            try:
                verification.append(VerificationTag(v))
            except ValueError:
                pass

        prior = _CATEGORY_PRIORS.get(primary.value, _CATEGORY_PRIORS[_FALLBACK_CATEGORY.value])
        if not verification:
            verification = list(prior["verification"])

        if primary in (TaskCategory.GENERATION, TaskCategory.DEBUGGING):
            for t in (VerificationTag.SYNTAX, VerificationTag.SYNTHESIZABILITY):
                if t not in verification:
                    verification.append(t)
        if primary == TaskCategory.OPTIMIZATION and VerificationTag.OPT_VALIDITY not in verification:
            verification.append(VerificationTag.OPT_VALIDITY)

        return TaskRequirements(
            raw_question          = question,
            title                 = title,
            category              = CategoryResult(primary, secondary),
            user_intent           = (data.get("user_intent") or question[:160]).strip(),
            required_output       = list(data.get("required_output") or []),
            verification          = verification,
            risk_profile          = _parse_risk(data.get("risk_profile"), prior["risk"]),
            answer_policy         = _parse_policy(data.get("answer_policy"), prior["policy"]),
            formatted_description = data.get("formatted_description", question[:200]),
            dataset_label         = dataset_label,
            classification_review = {
                "prior":  prior_cat,
                "review": review,
                "final":  primary.value,
                "reason": (data.get("review_reason") or "").strip(),
            },
        )

    def _fallback(self, title: str, question: str,
                  dataset_label: dict) -> TaskRequirements:
        prior_cat = _prior_category(dataset_label)
        if prior_cat is not None:
            primary, review = _coerce_category(prior_cat), "keep (fallback)"
        else:
            combined = title + " " + question
            scores = {
                TaskCategory.GENERATION:   len(self._CODE_KW.findall(combined)),
                TaskCategory.DEBUGGING:    len(self._DEBUG_KW.findall(combined)),
                TaskCategory.CONCEPTUAL:   len(self._CONC_KW.findall(combined)),
                TaskCategory.OPTIMIZATION: len(self._OPT_KW.findall(combined)),
            }
            primary = max(scores, key=lambda k: scores[k])
            if scores[primary] == 0:
                primary = _FALLBACK_CATEGORY
            review = "fresh (fallback)"
        prior = _CATEGORY_PRIORS[primary.value]
        return TaskRequirements(
            raw_question          = question,
            title                 = title,
            category              = CategoryResult(primary),
            user_intent           = question[:160],
            required_output       = [],
            verification          = list(prior["verification"]),
            risk_profile          = _parse_risk(None, prior["risk"]),
            answer_policy         = _parse_policy(None, prior["policy"]),
            formatted_description = f"[rule-fallback] {title}: {question[:150]}",
            dataset_label         = dataset_label,
            classification_review = {"prior": prior_cat, "review": review,
                                     "final": primary.value, "reason": ""},
        )


def _route_model(req: TaskRequirements) -> ModelConfig:
    cat  = req.category.primary
    risk = req.risk_profile
    m    = _cfg.models

    if cat == TaskCategory.GENERATION:
        return m.codegen
    if cat == TaskCategory.DEBUGGING or risk.consistency_risk == RiskLevel.HIGH:
        return m.consistency
    if cat == TaskCategory.CONCEPTUAL or risk.readability_risk == RiskLevel.HIGH:
        return m.readability
    if cat == TaskCategory.OPTIMIZATION or \
       RiskLevel.HIGH in (risk.redundancy_risk, risk.conciseness_risk):
        return m.concise
    return m.default


def _risk_directives(risk: RiskProfile) -> str:
    notes = []
    if risk.conciseness_risk == RiskLevel.HIGH:
        notes.append("Be concise: no padding, no restating the question, no length-for-richness.")
    if risk.redundancy_risk == RiskLevel.HIGH:
        notes.append("No redundant summaries or repeated explanations.")
    if risk.consistency_risk == RiskLevel.HIGH:
        notes.append("Do not guess: avoid hedging with several possible answers.")
    if risk.readability_risk == RiskLevel.HIGH:
        notes.append("Keep structure clean and signal/section names meaningful.")
    return "Directives: " + (" ".join(notes) if notes else "standard care.")


_M2_STRATEGY: dict[str, list[str]] = {
    TaskCategory.CONCEPTUAL.value: [
        "Produce: conclusion first + a short explanation + at most one small example.",
        "Limit background; do not expand into unrelated concepts.",
    ],
    TaskCategory.DEBUGGING.value: [
        "Produce: bug location + root cause + the MINIMAL fix code.",
        "Do not list many possible causes. If information is insufficient, state the single most "
        "likely cause and make your assumption explicit.",
    ],
    TaskCategory.GENERATION.value: [
        "Produce: the core Verilog code that meets the requirement + a brief explanation.",
        "Do NOT add a testbench by default. Do NOT provide multiple implementations. "
        "Do NOT add features the user did not request.",
    ],
    TaskCategory.OPTIMIZATION.value: [
        "Produce: optimization points + concrete change suggestions + an impact note.",
        "Focus on the relevant target among PPA / timing / area / power / readability. "
        "Avoid generic optimization theory.",
    ],
}

_M2_ROLE_HINT: dict[str, str] = {
    TaskCategory.CONCEPTUAL.value:   "You explain HDL concepts and simulation semantics precisely and briefly.",
    TaskCategory.GENERATION.value:   "You write synthesizable, requirement-faithful Verilog/SystemVerilog.",
    TaskCategory.DEBUGGING.value:    "You localise HDL bugs and give the minimal correct fix.",
    TaskCategory.OPTIMIZATION.value: "You improve HDL designs for PPA/timing/area/power/readability with concrete changes.",
}


def _build_m2_system(req: TaskRequirements) -> str:
    lines = [
        "You are a professional Verilog/SystemVerilog/VHDL engineer generating an answer DRAFT.",
        "",
        f"Title       : {req.title}",
        f"Question    : {req.raw_question}",
        f"User intent : {req.user_intent}",
    ]
    return "\n".join(lines)


def run_m2(req: TaskRequirements) -> str:
    system_msg = _build_m2_system(req)
    model_cfg  = _route_model(req)
    _substep(f"能力路由 → 模型 {_C.bold(model_cfg.model)}（任务={req.category.primary.value}）")
    logger.info("[M2] routed model=%s for task=%s",
                model_cfg.model, req.category.primary.value)

    assistant = AssistantAgent(
        name           = "M2_DraftGenerator",
        system_message = system_msg,
        llm_config = _llm_config_for(model_cfg, temperature=0.0),
    )
    proxy = UserProxyAgent(
        name             = "M2_Proxy",
        human_input_mode = "NEVER",
        max_consecutive_auto_reply = 1,
        is_termination_msg = lambda m: "DRAFT_DONE" in (m.get("content") or ""),
        code_execution_config = False,
    )

    user_msg = (
        f"Please generate an answer draft for the following requirement:\n\n"
        f"Title      : {req.title}\n\n"
        f"Question   : {req.raw_question}\n\n"
        f"Description: {req.formatted_description}\n\n"
        f"End your response with exactly one line: DRAFT_DONE"
    )
    proxy.initiate_chat(assistant, message=user_msg, silent=True)

    history = assistant.chat_messages.get(proxy, [])
    parts = [
        msg["content"] for msg in history
        if msg.get("role") == "assistant"
        and msg.get("content")
        and msg["content"].strip() != user_msg.strip()
    ]
    raw_reply = "\n".join(parts).strip()

    if not raw_reply:
        _block("M2 完整 LLM 回复", "[M2 produced no valid draft]")
        return "[M2 produced no valid draft]"

    _result_line(True, f"LLM 回复完成（{len(raw_reply)} 字符）")
    _block("M2 完整 LLM 回复（原始）", raw_reply)

    draft = raw_reply.replace("DRAFT_DONE", "").strip()
    _result_line(True, f"M2 最终草稿（剥离 DRAFT_DONE 后，{len(draft)} 字符）：")
    _block("M2 完整答案", draft)
    return draft


logger = logging.getLogger(__name__)

_CORE_OPEN = "⟦CORE⟧"
_CORE_CLOSE = "⟦/CORE⟧"


def _strip_core_markers(text: str) -> str:
    return (text or "").replace(_CORE_OPEN, "").replace(_CORE_CLOSE, "").strip()


# 各任务类别下"核心"的侧重（拆分时提示模型核心长什么样）
_EXTRACT_TYPE: dict[str, str] = {
    TaskCategory.DEBUGGING.value:
        "each distinct fix / repair point / diagnostic action that directly addresses the user's "
        "specific bug. Several independent fix points are SEVERAL cores",
    TaskCategory.GENERATION.value:
        "each implementation that directly fulfils the user's specific requirement; alternative "
        "independently-usable implementations are SEVERAL cores",
    TaskCategory.OPTIMIZATION.value:
        "each optimization approach that directly targets the user's specific design; independently "
        "applicable approaches are SEVERAL cores",
    TaskCategory.CONCEPTUAL.value:
        "each part that DIRECTLY and SPECIFICALLY answers what the user actually asked. Parts that "
        "directly resolve the user's specific question are cores; generic concept tours, broad "
        "background, generic 'when to use each' guidance, illustrative-only examples and "
        "point-by-point restatements of the user's own words are NON-core (context)",
}

# -----------------------------------------------------------------------------
# 拆分：高召回，按统一定义切核心 / 非核心
# -----------------------------------------------------------------------------

_M3_SPLIT_SYSTEM = (
    "You are a precise segmenter for Verilog/SystemVerilog/VHDL answer drafts.\n"
    "Split a draft into ordered segments and label each as CORE or CONTEXT, following ONE "
    "definition:\n"
    "  CORE    = a block that DIRECTLY and SPECIFICALLY answers the user's actual question / "
    "solves the user's actual problem.\n"
    "  CONTEXT = everything that does NOT directly solve it, namely: (1) generic guidance / best "
    "practices / 'when to use each' not tied to the user's specific case; (2) closing summaries, "
    "recaps, 'Summary', 'In conclusion', 'Key takeaway', comparison/recommendation tables, and "
    "their Chinese equivalents (总结 / 综上 / 小结 / 要点); (3) background, problem restatement, "
    "root-cause re-explanation; (4) loosely-related elaboration, illustrative-only examples, and "
    "point-by-point restatements of what the user already said.\n\n"
    "HOW MANY CORE SEGMENTS — RECALL FIRST:\n"
    "  - A draft OFTEN contains SEVERAL cores. Whenever the draft offers multiple distinct points "
    "that each directly address the user's problem — multiple fix points, multiple repair actions, "
    "multiple alternative solutions, multiple independently-usable approaches — emit ONE core "
    "segment PER point. Do this WHETHER OR NOT they are mutually exclusive: complementary fixes "
    "that must all be done, OR alternative solutions where the user picks one, are BOTH split into "
    "multiple cores. Selecting/pruning among them is decided LATER, not here.\n"
    "  - Multi-point structure is signalled by numbered/sectioned organisation (`1.`/`2.`, "
    "`### 1`/`### 2`, `Solution`/`Alternative`, `Option A/B`, `Fix 1/2`, `方法一/二`, bold "
    "headed paragraphs) AND by semantics (each block solving its own sub-problem). Rely on the "
    "SEMANTICS, not only on explicit 'Solution/Option' keywords.\n"
    "  - DEFAULT TO SPLITTING when unsure whether two points are separable: prefer MORE cores. "
    "Over-splitting is corrected downstream; missing a core is not.\n\n"
    "WHEN TO KEEP ONE CORE:\n"
    "  - A SINGLE solution described in sequential build steps ('first sketch the module, then "
    "fill the always block, then wire the output') is ONE core — these are construction steps of "
    "ONE thing, not separable solutions.\n"
    "  - A single explanation delivered in parts / bullets that together form one answer is ONE "
    "core.\n"
    "  Distinguish: fixes/approaches that target DIFFERENT sub-problems or DIFFERENT modules, or "
    "that the user could adopt independently -> SEVERAL cores; consecutive steps that build the "
    "SAME single artifact -> ONE core.\n\n"
    "ALWAYS emit at least one CORE. If the entire draft directly answers the question, output a "
    "single core (plus any context segments). Never output zero cores.\n"
    "Output only valid JSON — no markdown fences, no preamble."
)

_M3_SPLIT_PROMPT = """Split the following draft answer into ordered segments per the definition.

Task category: {category}
User question: {question}

For this category, a CORE block is: {core_desc}
CONTEXT is everything that does NOT directly/specifically solve the user's problem: generic
guidance, closing summaries/recaps/tables ("Summary"/"总结"/"综上"/"Key takeaway"), background /
problem restatement / root-cause re-explanation, and loosely-related elaboration or
illustrative-only examples.

Draft:
{draft}

Output a JSON array IN DOCUMENT ORDER. Each element:
{{
  "id":      "<s1, s2, s3, ...>",
  "role":    "<core|context>",
  "title":   "<short phrase for a core segment, else empty string>",
  "content": "<verbatim text of this segment, copied exactly>"
}}

Rules:
- Preserve original order; copy content verbatim (keep code blocks intact).
- RECALL FIRST: emit ONE core per distinct point that directly addresses the user's problem
  (multiple fixes / repairs / alternatives / approaches -> multiple cores), whether complementary
  or mutually exclusive. Do NOT merge several such points into one core. Pruning happens later.
- Decisive test for a core: "Does this block, on its own, directly and specifically address what
  the user asked?" If yes -> it is a core (its own segment). If it is generic guidance, a closing
  summary/table, background/restatement, or loosely-related elaboration -> context.
- Sequential build steps of ONE artifact, or one explanation split into bullets, stay as ONE core.
- Closing summaries, recaps, comparison/recommendation tables and 'Key takeaway'/'总结'/'综上'
  blocks are ALWAYS context, regardless of how many cores precede them.
- When unsure whether two points are separable, SPLIT into separate cores.
- Always output at least one core segment."""


def _split_core_context(draft: str, req: "TaskRequirements") -> list[dict]:
    core_desc = _EXTRACT_TYPE.get(req.category.primary.value,
                                  "the part that directly and specifically solves the request")
    prompt = _M3_SPLIT_PROMPT.format(
        category=req.category.primary.value,
        question=req.raw_question[:1500],
        core_desc=core_desc,
        draft=draft,
    )
    try:
        segs = json.loads(_strip_fences(
            _raw_llm(prompt, system=_M3_SPLIT_SYSTEM,
                     temperature=0.0, model_cfg=_cfg.models.m3_split)))
        if isinstance(segs, list) and segs:
            logger.info("M3 split draft into %d segment(s)", len(segs))
            return segs
    except Exception as exc:
        logger.warning("M3 core/context split failed (%s)", exc)
    return []


def _reassemble_m3(segments: list[dict], refined_core: str) -> str:
    """把精筛后的核心放回原位（包 ⟦CORE⟧），context 段原样保留供 M4 处理。"""
    core_block = f"{_CORE_OPEN}\n{refined_core.strip()}\n{_CORE_CLOSE}"
    pieces, core_done = [], False
    for s in segments:
        if s.get("role") == "core":
            if not core_done:
                pieces.append(core_block)
                core_done = True
        else:
            c = (s.get("content") or "").strip()
            if c:
                pieces.append(c)
    if not core_done:
        pieces.insert(0, core_block)
    return "\n\n".join(p for p in pieces if p)


# -----------------------------------------------------------------------------
# 群体辩论：所有核心候选一起审，在相互关系中取舍
# -----------------------------------------------------------------------------




_NO_COMPLETENESS_FAIL = (
"\n\nABSOLUTE RULES — NEVER eliminate a candidate merely for FORM differences:\n"
"  - It is terse / gives little or no explanation.\n"
"  - It does not restate, echo, verify or 'address' the user's exact wording.\n"
"  - It answers in a different wording style or presentation form.\n"
"  - It omits commentary / comparison / the 'full request'.\n"
"  - It endorses the user's existing approach as correct.\n"
"A correct, usable, on-topic block that simply SAYS LESS, or differs only in FORM, is KEPT.\n"
"\n"
"However, FORM difference is not the same as an independent core. A candidate may still be "
"ELIMINATED when it is only a wrapper, parameter-only variant, scope-only variant, checklist "
"item, or weaker duplicate of another candidate. Verbosity and style are never grounds for "
"elimination; lack of independent technical value can be."
)

_M3_GROUP_SYSTEM = (
"You are a panel of senior Verilog/SystemVerilog/VHDL/Vivado/ModelSim reviewers performing a "
"SINGLE group review over ALL core answer candidates of one draft AT ONCE.\n"
"Each candidate is a block that purports to directly address the user's problem. Looking at "
"the candidates TOGETHER and in their mutual relationships, decide which to KEEP and which to "
"ELIMINATE, with a reason for each.\n\n"

"FIRST, identify THE USER'S ACTUAL QUESTION/GOAL — the specific problem the user asked to "
"solve. Judge every candidate against THAT goal.\n\n"

"KEEP a candidate when it is technically correct, usable, on-topic for the user's actual "
"question, respects the user's explicitly stated hard constraints, AND provides independent "
"technical value as a core answer.\n\n"

"In particular, KEEP candidates that are:\n"
"  - COMPLEMENTARY fixes that must all be applied;\n"
"  - independent confirmed defects visible in the user's posted code;\n"
"  - genuinely different mechanisms for solving the same problem;\n"
"  - independently valid alternatives that a user could choose between, where each alternative "
"has a distinct implementation mechanism or distinct tradeoff.\n\n"

"Multiple correct cores should ALL be kept — there is NO fixed limit. But do not count a "
"minor variant, wrapper, parameter change, scope change, or checklist note as a separate core.\n\n"

"ELIMINATE a candidate ONLY for one of these concrete reasons:\n"
"  1. TECHNICAL ERROR — wrong HDL semantics, broken / non-synthesizable code, invalid command, "
"or a diagnosis that contradicts the symptom.\n"
"\n"
"  2. INFERIOR / REDUNDANT / DERIVATIVE ALTERNATIVE — it targets the SAME sub-problem as a "
"better candidate and is strictly weaker, subsumed, or derivative. This includes:\n"
"     - WRAPPER-ONLY VARIANT: e.g. a function that merely wraps the same case statement already "
"kept elsewhere, without adding a genuinely parameterized or safer mechanism.\n"
"     - PARAMETER-ONLY VARIANT: e.g. the same command/API/algorithm with only a different "
"argument, scope, hierarchy, formatting, or limit.\n"
"     - SCOPE-ONLY VARIANT: e.g. `log -r sim:/testbench/*` when `log -r /*` already states the "
"core ModelSim mechanism; the selective version is only an implementation note.\n"
"     - WEAKER DUPLICATE: same diagnosis or same fix with less complete, less direct, or less "
"reliable code.\n"
"     Keep the clearest candidate that states the core mechanism; eliminate the derivative "
"candidate and mention which candidate subsumes it.\n"
"\n"
"  3. GENERIC GUIDANCE — boilerplate best-practice NOT tied to the user's specific code or "
"problem. This includes generic checklist items, broad debugging advice, or low-information "
"commands that do not identify a specific likely cause or core solution for THIS user.\n"
"\n"
"  4. VIOLATES A USER HARD CONSTRAINT — the candidate breaks a limitation the user EXPLICITLY "
"wrote about the SOLUTION ITSELF, such as 'single always block', 'do not change the interface', "
"'must be synthesizable', 'no vendor primitives', or 'avoid separate blocks', and does the very "
"thing the user ruled out, WITHOUT being the unavoidable best answer under that constraint.\n"
"\n"
"  5. OFF-TOPIC EXTRA — the candidate turns to solve a DIFFERENT problem the user did NOT ask "
"about, even if that problem is real in the user's code and the advice is technically correct. "
"It is extra advice aimed at a goal the user did not raise.\n"
"\n"
"  6. SPECULATIVE LONG-SHOT — applies when the user reports a symptom or partial code and the "
"draft lists several guessed causes. Among guessed causes, KEEP only the most probable, "
"question-specific root causes. ELIMINATE low-probability scattershot possibilities, especially "
"generic FPGA/HDL debugging guesses such as:\n"
"     - 'check synthesis optimization' when the signal is already used or no evidence suggests "
"optimization is the root cause;\n"
"     - 'check vector width mismatch' when the shown declarations already match or the issue is "
"not causally explained by width;\n"
"     - 'dump a VCD', 'check the testbench', 'check warnings', or similar checklist advice when "
"not tied to a specific failure mode;\n"
"     - any 'maybe also' branch that is merely possible but not one of the strongest diagnoses.\n"
"\n"
"     HARD GUARD — reason 6 NEVER applies when the candidates are CONFIRMED, independent defects "
"visible in code the user DID provide, such as a syntax error, an out-of-range index, an illegal "
"dynamic slice, a wrong keyword, or a missing required assignment each sitting in the posted "
"source. Those are complementary fixes that must ALL be kept, no matter how many. The test is: "
"GUESS about an unseen or weakly evidenced fault -> prunable under reason 6; CONFIRMED defect "
"in shown code -> KEEP.\n\n"

"CRITICAL — distinguish INDEPENDENT ALTERNATIVES from DERIVATIVE VARIANTS:\n"
"  - Independent alternative = solves the same user goal through a genuinely different mechanism "
"or materially different implementation strategy. KEEP it.\n"
"    Example: replacing a dynamic VHDL slice with a static case mux vs replacing it with a static "
"for-loop mux. These are distinct synthesizable implementation strategies, so both may be kept.\n"
"  - Derivative variant = same mechanism with only a wrapper, parameter, scope, formatting, or "
"minor organization change. ELIMINATE it as redundant.\n"
"    Example: a function wrapping the same hard-coded case mux is not an independent solution if "
"the direct case mux is already kept.\n"
"    Example: selective ModelSim logging is not a separate core if the main candidate already "
"explains using `log` before `run`; it is only a scope parameter note.\n\n"

"CRITICAL — how to apply reason 5 WITHOUT over-pruning:\n"
"  - The test is NOT 'is this candidate strictly necessary?'. Necessity alone is the wrong test "
"and would wrongly cut useful on-topic supplements.\n"
"  - The test IS: 'Is this candidate still answering THE USER'S ACTUAL QUESTION with independent "
"technical value, or has it switched to solving a different unasked goal?'\n"
"  - A candidate that is causally part of fixing the user's actual problem is NOT off-topic, "
"even if it touches another module. Only eliminate when its PURPOSE is a different goal.\n"
"  - When unsure whether it is the same goal or a different goal -> KEEP, unless it is clearly "
"a derivative variant under reason 2 or a speculative long-shot under reason 6.\n\n"

"CRITICAL — distinguish a HARD CONSTRAINT from a FORM difference:\n"
"  - HARD CONSTRAINT = an explicit limitation the user wrote on the solution itself "
"(structure, interface, technology, synthesizability).\n"
"  - FORM difference = terseness, wording, presentation style, or omitting explanation. NEVER "
"grounds for elimination.\n"
"  - A candidate that honestly shows a constraint is impossible and gives the closest valid "
"solution under it is KEPT.\n\n"

"Do NOT eliminate two candidates as duplicates when they address DIFFERENT sub-problems, are "
"complementary, or are genuinely independent mechanisms. DO eliminate candidates that only "
"repackage, narrow, wrap, or checklist the same already-kept mechanism."
+ _NO_COMPLETENESS_FAIL +
"\n\nOutput ONLY valid JSON — no markdown fences, no preamble."
)

_M3_GROUP_PROMPT = """Review ALL core candidates for one HDL answer together and decide keep/eliminate.

Task category : {category}
User question : {question}
Requirement   : {description}

STEP 1 — Identify THE USER'S ACTUAL QUESTION/GOAL: the specific problem the user asked to solve.
Judge every candidate against THAT goal.

Candidates:
{candidates_block}

STEP 2 — Judge the candidates TOGETHER, including their mutual relationships.

KEEP every candidate that is:
- technically correct;
- usable;
- on-topic for the user's actual question;
- compliant with explicit user hard constraints;
- and independently valuable as a core answer.

KEEP complementary fixes, confirmed independent defects visible in posted code, and genuinely
different implementation mechanisms. There is NO fixed limit on how many candidates may be kept.

ELIMINATE a candidate ONLY for one of these reasons:

(1) TECHNICAL ERROR:
    It has wrong HDL semantics, broken code, non-synthesizable code where synthesizability is
    required, an invalid command, or a diagnosis that contradicts the symptom.

(2) INFERIOR / REDUNDANT / DERIVATIVE ALTERNATIVE:
    It targets the SAME sub-problem as a better candidate and is strictly weaker, subsumed, or
    derivative. This includes:
    - wrapper-only variants;
    - parameter-only variants;
    - scope-only variants;
    - formatting-only or organization-only variants;
    - weaker duplicates of another kept candidate.

    Examples:
    - A function that merely wraps the same hard-coded case statement should be eliminated if the
      direct case-statement solution is already kept.
    - `log -r sim:/testbench/*` should be eliminated as an independent core if another candidate
      already explains the core ModelSim method: use `log` before `run`. The selective version is
      only a scope/memory note.
    - A candidate that only says "use a smaller range", "adjust the bound", or "same command with
      another hierarchy" is not independent unless it introduces a materially different mechanism.

(3) GENERIC GUIDANCE:
    It is boilerplate best-practice or checklist advice not tied to the user's specific code,
    symptom, or requested goal.

(4) VIOLATES A USER HARD CONSTRAINT:
    It breaks an explicit limitation the user wrote about the solution itself, when respecting
    that limitation was possible.

(5) OFF-TOPIC EXTRA:
    It turns to solve a DIFFERENT problem the user did NOT ask about, even if that advice is
    technically correct.

(6) SPECULATIVE LONG-SHOT:
    The user gave only a symptom or partial evidence, the draft lists several guessed causes, and
    this candidate is a low-probability scattershot possibility rather than one of the strongest,
    question-specific root causes.

    Examples of candidates that should usually be eliminated under this reason:
    - synthesis optimization guesses when the signal is already consumed or there is no evidence
      optimization is the root cause;
    - vector width mismatch guesses when shown widths already match or the symptom is not explained
      by width;
    - generic "check warnings", "dump VCD", "check testbench", "use slower clock" advice when not
      tied to a specific likely failure mode.

HARD GUARDS:
- Do NOT apply reason (6) to confirmed, independent defects visible in code the user actually
  posted. Confirmed defects are complementary fixes and should be kept.
- Do NOT eliminate for FORM differences: terseness, different wording, different presentation form,
  omitting explanation, or endorsing the user's existing approach.
- Do NOT use "strict necessity" as the elimination test. A useful on-topic supplement may be kept
  if it has independent technical value.
- But DO eliminate derivative variants that only wrap, narrow, parameterize, or restate an already
  kept mechanism.

Output ONLY a JSON object with this schema:
{{
  "user_goal": "<one sentence: the specific problem the user actually asked to solve>",
  "candidates": [
    {{
      "id": "<candidate id>",
      "decision": "<keep|eliminate>",
      "reason": "<one sentence. For eliminate, explicitly name the reason category and, if redundant, say which better candidate subsumes it.>"
    }}
  ],
  "panel_reason": "<one short sentence on the overall keep/eliminate rationale>"
}}

Rules:
- Include EVERY candidate id exactly once.
- There is NO limit on how many you keep.
- Keep all correct, usable, distinct, on-topic, constraint-respecting candidates with independent
  technical value.
- Eliminate only on grounds (1)-(6) above; never on form/verbosity grounds.
"""
def _norm_id(x) -> str:
    return str(x or "").strip().strip("[]").strip()

def _verify_all_candidates(candidates: list[dict],
                           req: "TaskRequirements") -> tuple[list[dict], dict, list[dict]]:
    """一次群体辩论：返回 (保留的候选, {id: 淘汰理由}, 每个候选的判定记录)。"""
    if len(candidates) <= 1:
        recs = [{"item_id": c["id"], "title": c.get("title", ""),
                 "content": c.get("content", ""), "decision": "keep",
                 "passed": True, "reason": "single core candidate — kept"}
                for c in candidates]
        return candidates, {}, recs

    candidates_block = "\n\n".join(
        f"### Candidate\nid: {c['id']}\ntitle: {c.get('title') or c['id']}\ncontent:\n{c.get('content', '')}"
        for c in candidates
    )

    prompt = _M3_GROUP_PROMPT.format(
        category=req.category.primary.value,
        question=req.raw_question[:1000],
        description=req.formatted_description[:600],
        candidates_block=candidates_block,
    )

    by_id = {_norm_id(c["id"]): c for c in candidates}
    decisions: dict[str, dict] = {}

    try:
        data = json.loads(_strip_fences(
            _raw_llm(prompt, system=_M3_GROUP_SYSTEM,
                     temperature=0.0, model_cfg=_cfg.models.m3)
        ))

        for entry in (data.get("candidates") or []):
            cid = _norm_id(entry.get("id"))
            if cid in by_id:
                decisions[cid] = {
                    "decision": (entry.get("decision") or "keep").strip().lower(),
                    "reason": (entry.get("reason") or "").strip(),
                }

        missing = set(by_id) - set(decisions)
        if missing:
            logger.warning("M3 group debate missing candidate ids after normalization: %s",
                           sorted(missing))

        logger.info("M3 group debate: %s (panel: %s)",
                    {k: v["decision"] for k, v in decisions.items()},
                    data.get("panel_reason", ""))

    except Exception as exc:
        logger.warning("M3 group debate failed (%s) — keeping all candidates", exc)

    kept, elim_notes, records = [], {}, []

    for c in candidates:
        cid = _norm_id(c["id"])
        d = decisions.get(cid, {"decision": "keep", "reason": "no panel verdict — kept"})
        is_keep = d["decision"] != "eliminate"

        records.append({
            "item_id": c["id"],
            "title": c.get("title", ""),
            "content": c.get("content", ""),
            "decision": "keep" if is_keep else "eliminate",
            "passed": is_keep,
            "reason": d["reason"],
        })

        if is_keep:
            kept.append(c)
        else:
            elim_notes[c["id"]] = d["reason"] or "群体辩论判定淘汰"

    if not kept:
        logger.warning("M3 group debate eliminated ALL candidates — falling back to keep all")
        for r in records:
            r["decision"], r["passed"] = "keep", True
            r["reason"] = "fallback: panel eliminated all, kept"
        return candidates, {}, records

    return kept, elim_notes, records


def _assemble_verified_answer(passed_candidates: list[dict],
                              req: "TaskRequirements") -> str:
    if len(passed_candidates) == 1:
        return passed_candidates[0]["content"].strip()

    items_text = "\n\n".join(
        f"### [{c['id']}] {c.get('title', '')}\n{c['content']}" for c in passed_candidates
    )
    prompt = (
        f"The following verified items are the core answers to a Verilog/HDL question.\n"
        f"Task category: {req.category.primary.value}\n"
        f"User question: {req.raw_question[:800]}\n\n"
        f"Verified items:\n{items_text}\n\n"
        "Assemble these items into a single coherent answer. Rules:\n"
        "1. Do NOT alter the technical content of any item.\n"
        "2. Add only minimal transition sentences where needed.\n"
        "3. Preserve all code blocks exactly as given.\n"
        "4. Do not add new information or commentary not present in the items.\n"
        "Output the assembled answer only. End with: ASSEMBLE_DONE"
    )
    try:
        return _raw_llm(prompt, temperature=0.0,
                        model_cfg=_cfg.models.m3).replace("ASSEMBLE_DONE", "").strip()
    except Exception as exc:
        logger.warning("Assembly LLM call failed (%s) — concatenating items", exc)
        return "\n\n---\n\n".join(c["content"] for c in passed_candidates)


# -----------------------------------------------------------------------------
# run_m3：拆分(高召回) -> 一次群体辩论 -> 重组保留核心
# -----------------------------------------------------------------------------

def run_m3(req: "TaskRequirements", draft: str) -> tuple[str, list[dict]]:
    feedback_log: list[dict] = []

    _substep("拆分核心 / 非核心（高召回）…")
    segments = _split_core_context(draft, req)
    core_segs = [s for s in segments if s.get("role") == "core"]
    context_segs = [s for s in segments if s.get("role") != "core"]
    if not core_segs:
        segments = [{"id": "core_1", "role": "core", "title": "Full draft",
                     "content": draft}]
        core_segs, context_segs = segments, []

    total = len(core_segs)
    _result_line(True, f"核心候选 {_C.bold(str(total))} 个 / 非核心段 {len(context_segs)} 个")
    for c in core_segs:
        _detail(f"• 核心 [{c['id']}] {c.get('title', '')}")

    candidates = [{"id": s["id"], "title": s.get("title") or s["id"],
                   "content": s.get("content", "")} for s in core_segs]

    # ---- 单核心：直接透传 ----
    if len(candidates) <= 1:
        _substep(_C.cyan("仅 1 个核心候选 —— 跳过群体辩论，原样透传"))
        cand = candidates[0] if candidates else {"id": "core_1", "title": "", "content": draft}
        feedback_log.append({
            "attempt": 1, "item_id": cand["id"], "title": cand.get("title", ""),
            "passed": True, "decision": "keep", "skipped": True,
            "content": cand.get("content", ""),
            "feedback": "[M3 skipped] single core candidate — passed through unchanged",
            "reason": "single core candidate",
        })
        return _reassemble_m3(segments, cand["content"].strip()), feedback_log

    # ---- 多核心：一次群体辩论 ----
        # ---- 多核心：一次群体辩论 ----
    _substep(f"群体辩论：{total} 个核心候选一起审…")
    for c in candidates:
            _block(f"核心候选 [{c['id']}] {c.get('title', '')}", c.get("content", ""))

    if _cfg.ablation_mode == "no_debate":
            # 消融：跳过群体辩论，所有 core 候选全部保留（不剪枝）
            _substep(_C.yellow("[ablation:no_debate] 跳过群体辩论 —— 全部核心保留"))
            kept = list(candidates)
            elim_notes = {}
            records = [{"item_id": c["id"], "title": c.get("title", ""),
                        "content": c.get("content", ""), "decision": "keep",
                        "passed": True, "reason": "ablation:no_debate — all kept"}
                       for c in candidates]
    else:
            kept, elim_notes, records = _verify_all_candidates(candidates, req)

    for r in records:
        r["attempt"] = 1
        r["feedback"] = f"[group] {r['decision']}: {r.get('reason', '')}"
        feedback_log.append(r)
        _result_line(r["passed"], f"[{r['item_id']}] "
                     + (_C.green("保留") if r["passed"] else _C.red("淘汰"))
                     + (f" —— {r.get('reason', '')}" if not r["passed"] else ""))

    sel_ids = {c["id"] for c in kept}
    for r in feedback_log:
        if "item_id" in r:
            r["m3_selected"] = r["item_id"] in sel_ids

    _result_line(True, f"核心保留 {_C.bold(str(len(kept)))}/{total} 个："
                 + ", ".join(_C.green(c["id"]) for c in kept))

    refined_core = _assemble_verified_answer(kept, req)
    return _reassemble_m3(segments, refined_core), feedback_log


# =============================================================================
# M4 —— 四类区分版（keep / compress / delete 三档处置）
# =============================================================================
# 设计依据（四类讨论定稿）:
#   M4 只处理 context（非核心）；core 由 M3 用 ⟦CORE⟧ 标记，M4 不碰。
#   三档处置（替代原 keep/delete 二分）:
#     keep      —— 原样保留（该类别的“招牌段”，用户真正想要的，绝不删/压）
#     compress  —— 压缩：去水分/去重复，但所有信息点全保留
#     delete    —— 删除（纯回声/收尾总结/与本题无关的发挥）
#   四类“招牌段”各自保命:
#     debugging   —— 几乎全回声，最敢删；只有真正相关的才压
#     conceptual  —— 表格/示例可能带 core 没覆盖的信息 → 谨慎，偏压不删
#     optimization—— 权衡/缺点分析(opt_problem/opt_impact)=用户要的 → keep，绝不 delete
#     generation  —— 实现坑/边界条件(usage_constraints/design_notes)=保命 → keep
#   跨四类共性: 收尾总结/summary 一律可 delete。
# =============================================================================

_M4_LAYOUT: dict[str, dict] = {
    TaskCategory.CONCEPTUAL.value: {
        "order": ["conclusion", "explanation", "example"],
        # 概念类：非核心的收尾结论→删；铺垫解释、示例可能带信息→压（谨慎不删）
        "disposition": {
            "conclusion":  "delete",    # 收尾重述型结论
            "explanation": "compress",  # 铺垫性解释（核心解释已被 M3 标走）
            "example":     "compress",  # 示例/对比：可能带 core 没覆盖的信息，压不删
        },
    },
    TaskCategory.DEBUGGING.value: {
        "order": ["bug_cause", "fix_approach", "code", "fix_rationale"],
        # 调错类：context 几乎全回声，最敢删/压；未标记代码保留
        "disposition": {
            "bug_cause":     "compress",  # 现象/根因复述（核心诊断已被 M3 标走）
            "fix_approach":  "compress",  # 非核心的方法补充
            "code":          "keep",      # 若有未被标记的代码块，保留
            "fix_rationale": "compress",  # 修复理据
        },
    },
    TaskCategory.GENERATION.value: {
        "order": ["code", "design_notes", "usage_constraints"],
        # 生成类：实现坑/边界条件保命（design_notes / usage_constraints）
        "disposition": {
            "code":              "keep",  # 未被标记的代码，保留
            "design_notes":      "keep",  # 实现注意/设计权衡：招牌段，保命
            "usage_constraints": "keep",  # 用法约束/边界坑：招牌段，保命
        },
    },
    TaskCategory.OPTIMIZATION.value: {
        "order": ["opt_problem", "opt_suggestion", "opt_example", "opt_impact"],
        # 优化类：权衡/缺点分析(opt_problem/opt_impact)=用户要的，保命，绝不 delete
        "disposition": {
            "opt_problem":    "keep",      # 缺点/问题分析：招牌段，保命
            "opt_suggestion": "keep",      # 方案（多为 core，漏标也保）
            "opt_example":    "compress",  # 示例：压
            "opt_impact":     "keep",      # 影响/权衡分析：招牌段，保命
        },
    },
}

# 消融用：不分 task 的统一处置表（no_taskaware）。
# 故意不体现 task-aware 的精细保护：full 版中 generation 的 design_notes /
# usage_constraints、optimization 的 opt_problem / opt_impact 都是 keep（招牌段保命），
# 这里一律降级为 compress，以体现 task-aware 保护了"该保的内容"。
_UNIFORM_DISPOSITION: dict[str, str] = {
    "conclusion":        "delete",
    "explanation":       "compress",
    "example":           "compress",
    "bug_cause":         "compress",
    "fix_approach":      "compress",
    "code":              "keep",      # 代码始终保留，与 full 一致
    "fix_rationale":     "compress",
    "design_notes":      "compress",  # full 版为 keep
    "usage_constraints": "compress",  # full 版为 keep
    "opt_problem":       "compress",  # full 版为 keep
    "opt_suggestion":    "compress",  # full 版为 keep
    "opt_example":       "compress",
    "opt_impact":        "compress",  # full 版为 keep
    "background":        "compress",
    "summary":           "delete",
    "other":             "compress",
}


# 额外类型（_M4_EXCESS_MENU 对应）默认处置 —— 跨四类共性：summary 可删。
# 仅覆盖真实存在的 excess 类型: background / testbench / alternative / summary / other
_EXCESS_DISPOSITION: dict[str, str] = {
    "background":  "compress",  # 背景铺垫：压（可能带少量信息）
    "summary":     "delete",    # 收尾总结/重述：删
    "testbench":   "keep",      # 受 answer_policy 门控，下方逻辑处理
    "alternative": "keep",      # 受 answer_policy 门控，下方逻辑处理
    "other":       "compress",  # 未知：保守压，不直接删
}

# answer_policy 门控类型：策略关闭时强制 delete，开启时按表（keep）
_POLICY_GATED = {"testbench": "include_testbench",
                 "alternative": "list_multiple_options"}

_CONTENT_TYPE_DESC: dict[str, str] = {
    "conclusion":        "the direct answer / conclusion to the question",
    "explanation":       "explanation of the concept or the reasoning",
    "example":           "a small illustrative example",
    "bug_cause":         "the root cause of the bug",
    "fix_approach":      "the strategy of the fix",
    "code":              "the core / corrected Verilog code",
    "fix_rationale":     "a brief note on why the fix works",
    "design_notes":      "key design notes about the code",
    "usage_constraints": "usage constraints or assumptions for the code",
    "opt_problem":       "the current problem in the design",
    "opt_suggestion":    "the optimization suggestion",
    "opt_example":       "the concrete modification example / code",
    "opt_impact":        "impact on timing / area / power / readability",
    "background":        "general background not specific to this question",
    "testbench":         "a testbench / verification stimulus",
    "alternative":       "an alternative or additional candidate solution",
    "summary":           "a generic recap or restated summary",
    "other":             "anything that matches none of the labels above",
}

_M4_EXCESS_MENU = ["background", "testbench", "alternative", "summary", "other"]

_M4_DECOMPOSE_SYSTEM = (
    "You are a precise content segmenter for Verilog/SystemVerilog/VHDL answers.\n"
    "Your only task: split an answer into its constituent content blocks and label each "
    "block's content type. You never rewrite, summarise, or invent content.\n"
    "Output only valid JSON — no markdown fences, no preamble."
)

_M4_DECOMPOSE_PROMPT = """Segment the following HDL answer into content blocks and label each block.

Task category : {category}
User intent   : {intent}

Answer to segment:
{answer}

Valid content_type labels for this task:
{type_menu}

Output a JSON array; each element must have exactly these fields:
{{
  "id":           "<b1, b2, b3, ...>",
  "content_type": "<one label from the list above; use 'other' if none fits>",
  "content":      "<the verbatim text of this block, copied exactly from the answer>"
}}

Rules:
- Cover the ENTIRE answer: every sentence and code block belongs to exactly one block.
- Keep each code block intact inside a single block; do not split code.
- Do NOT rewrite, paraphrase, summarise, merge unrelated content, or add anything.
- Copy the content field verbatim."""


def _decompose_answer(answer: str, req: TaskRequirements) -> list[dict]:
    layout    = _M4_LAYOUT[req.category.primary.value]
    menu_keys = list(layout["order"]) + _M4_EXCESS_MENU
    type_menu = "\n".join(f"- {k}: {_CONTENT_TYPE_DESC[k]}" for k in menu_keys)
    prompt = _M4_DECOMPOSE_PROMPT.format(
        category  = req.category.primary.value,
        intent    = req.user_intent,
        answer    = answer,
        type_menu = type_menu,
    )
    try:
        blocks = json.loads(_strip_fences(
            _raw_llm(prompt, system=_M4_DECOMPOSE_SYSTEM,
                     temperature=0.0, model_cfg=_cfg.models.m4)))
        if isinstance(blocks, list) and blocks:
            logger.info("M4 decomposed segment into %d content block(s)", len(blocks))
            return blocks
    except Exception as exc:
        logger.warning("M4 decomposition failed (%s)", exc)
    return []


def _disposition_for(content_type: str, req: TaskRequirements) -> str:
    """返回某 content_type 的处置: keep | compress | delete。"""
    # ---- 消融：no_compress —— 所有 non-core 原样保留，不删不压 ----
    if _cfg.ablation_mode == "no_compress":
        return "keep"

    ct = (content_type or "other").strip()
    category = req.category.primary.value

    # 1) answer_policy 门控类型优先（消融下仍生效，保持与 full 的策略一致性）
    if ct in _POLICY_GATED:
        enabled = getattr(req.answer_policy, _POLICY_GATED[ct], False)
        return "keep" if enabled else "delete"

    # ---- 消融：no_taskaware —— 所有类别用统一处置表（不按 category 选 layout）----
    if _cfg.ablation_mode == "no_taskaware":
        return _UNIFORM_DISPOSITION.get(ct, "compress")

    # 2) 类别专属处置表（full）
    disp = _M4_LAYOUT.get(category, {}).get("disposition", {})
    if ct in disp:
        return disp[ct]

    # 3) 额外类型默认表
    if ct in _EXCESS_DISPOSITION:
        return _EXCESS_DISPOSITION[ct]

    # 4) 未知类型：保守 compress
    return "compress"


# ---- 压缩：把所有 compress 块一次性交给 LLM，去水分留信息点 ----
_M4_COMPRESS_SYSTEM = (
    "You are a technical editor compressing NON-CORE (context) blocks of an HDL answer. "
    "Your ONLY job is to remove water — redundancy, filler, hedging, repeated restatements — while "
    "preserving EVERY genuine information point. You must NOT delete information, add information, "
    "change technical claims, or touch code inside the block (keep code verbatim). The result must "
    "say the SAME things with fewer words. If a block is already tight, return it nearly unchanged.\n"
    "Output ONLY valid JSON — no markdown fences, no preamble, no trailing text."
)

_M4_COMPRESS_PROMPT = """Compress the following NON-CORE context blocks of a {category} HDL answer.

User intent: {intent}

For EACH block, produce a compressed version that:
- keeps every real information point (facts, caveats, constraints, trade-offs, numbers, signal/port names),
- removes redundancy, filler, restatement, and hedging,
- keeps any code/commands EXACTLY as-is (do not rewrite code),
- does NOT add new claims and does NOT change technical meaning.

Blocks:
{blocks_block}

Return ONLY a JSON array, one object per block, in the SAME order:
[
  {{"idx": <int, matching the block index above>, "compressed": "<the compressed text>"}}
]"""


def _compress_blocks(blocks: list[dict], req: TaskRequirements) -> dict[int, str]:
    """对一批 compress 块整体压缩。返回 {block_index_in_blocks: compressed_text}。
    失败回退空 dict（调用方保留原文）。"""
    if not blocks:
        return {}
    lines = []
    for i, b in enumerate(blocks):
        content = (b.get("content") or "").strip()
        ct = (b.get("content_type") or "other").strip()
        lines.append(f'[block {i}] content_type="{ct}"\n{content}')
    prompt = _M4_COMPRESS_PROMPT.format(
        category=req.category.primary.value,
        intent=req.user_intent,
        blocks_block="\n\n".join(lines),
    )
    try:
        resp = _raw_llm(prompt, system=_M4_COMPRESS_SYSTEM,
                        temperature=0.0, model_cfg=_cfg.models.m4)
        data = json.loads(_strip_fences(resp))
        out: dict[int, str] = {}
        if isinstance(data, list):
            for item in data:
                try:
                    idx = int(item.get("idx"))
                except (TypeError, ValueError):
                    continue
                comp = (item.get("compressed") or "").strip()
                if comp:
                    out[idx] = comp
        return out
    except Exception as exc:
        logger.warning("M4 compression failed (%s) — keeping originals", exc)
        return {}


def _process_blocks(blocks: list[dict], req: TaskRequirements
                    ) -> tuple[str, list[dict], list[dict], list[dict]]:
    """对一组已 decompose 的块做三档处置。
    返回 (screened_text, kept_blocks, removed_blocks, compressed_records)。
      kept_blocks        = keep + compress 后保留进正文的块（用于模块计数）
      removed_blocks     = delete 的块（→ deleted_blocks）
      compressed_records = [{content_type, before, after}]（→ compressed_blocks）
    """
    # 先挑出 compress 块整体压缩
    comp_b = [b for b in blocks if _disposition_for(b.get("content_type"), req) == "compress"]
    comp_map = _compress_blocks(comp_b, req) if comp_b else {}
    # 建立 compress 块 → 其在 comp_b 中下标 的映射（按出现顺序）
    comp_idx_seq = iter(range(len(comp_b)))
    comp_idx_for = {}
    ci = 0
    for b in blocks:
        if _disposition_for(b.get("content_type"), req) == "compress":
            comp_idx_for[id(b)] = ci
            ci += 1

    out_parts, kept_blocks, removed, compressed_records = [], [], [], []
    for b in blocks:
        ct = (b.get("content_type") or "other").strip()
        disp = _disposition_for(ct, req)
        content = (b.get("content") or "").strip()
        if not content:
            continue
        if disp == "delete":
            removed.append(b)
            continue
        if disp == "compress":
            idx = comp_idx_for.get(id(b))
            after = comp_map.get(idx, content) if idx is not None else content
            after = (after or "").strip() or content
            compressed_records.append({"content_type": ct, "before": content, "after": after})
            out_parts.append(after)
            kb = dict(b); kb["content"] = after; kb["_disp"] = "compress"
            kept_blocks.append(kb)
        else:  # keep
            out_parts.append(content)
            kb = dict(b); kb["_disp"] = "keep"
            kept_blocks.append(kb)

    screened_text = "\n\n".join(p for p in out_parts if p.strip())
    return screened_text, kept_blocks, removed, compressed_records


def _screen_segment(segment_text: str, req: TaskRequirements
                    ) -> tuple[str, list[dict], list[dict], list[dict]]:
    """三档筛检（有标记路径用，处理 pre/post 非核心段）。"""
    seg = (segment_text or "").strip()
    if not seg:
        return "", [], [], []
    blocks = _decompose_answer(seg, req)
    if not blocks:
        return seg, [], [], []      # 拆解失败：整段保留，不计入统计
    return _process_blocks(blocks, req)


def _screen_without_markers(text: str, req: TaskRequirements
                            ) -> tuple[str, list[dict], list[dict], list[dict]]:
    """三档筛检（无标记路径）。"""
    blocks = _decompose_answer(text, req)
    if not blocks:
        return _strip_core_markers(text), [], [], []
    screened, kept, removed, comp = _process_blocks(blocks, req)
    if not screened:
        screened = _strip_core_markers(text)
    return screened, kept, removed, comp


def run_m4(m3_output: str, req: TaskRequirements) -> tuple[str, dict]:
    text          = (m3_output or "").strip()
    stripped_full = _strip_core_markers(text)
    metrics = {"removed_count": 0,
               "blocks_before": 0,        # ← 拆解前模块总数(含 core 计 1)
               "blocks_after":  0,        # ← 筛检后保留模块数(含 core 计 1)
               "blocks_removed": 0,       # ← 删除模块数(= removed_count)
               "compressed_count": 0,     # ← 压缩模块数
               "len_before": len(stripped_full),
               "len_after":  len(stripped_full),
               "reduction_ratio": 0.0,
               "deleted_blocks": [],
               "compressed_blocks": []}
    if not text:
        return text, metrics

    _substep(f"内容拆解模型（四类处置） → {_C.bold(_cfg.models.m4.model)}")

    def _finish(final: str, kept_count: int, removed_blocks: list,
                compressed_records: list) -> tuple[str, dict]:
        lb = metrics["len_before"]
        removed_count = len(removed_blocks)
        comp_count    = len(compressed_records)
        metrics["removed_count"]    = removed_count
        metrics["blocks_removed"]   = removed_count
        metrics["compressed_count"] = comp_count
        metrics["blocks_after"]     = kept_count
        metrics["blocks_before"]    = kept_count + removed_count
        metrics["len_after"]        = len(final)
        metrics["reduction_ratio"]  = (lb - len(final)) / lb if lb else 0.0
        metrics["deleted_blocks"]   = [
            {"content_type": b.get("content_type", "other"),
             "content":      (b.get("content") or "").strip()}
            for b in removed_blocks if (b.get("content") or "").strip()
        ]
        metrics["compressed_blocks"] = [
            {"content_type": r.get("content_type", "other"),
             "before":       (r.get("before") or "").strip(),
             "after":        (r.get("after") or "").strip()}
            for r in compressed_records if (r.get("before") or "").strip()
        ]
        return final, metrics

    # ---- 路径 2:无核心标记 ----
    if _CORE_OPEN not in text or _CORE_CLOSE not in text:
        _result_line(False, _C.yellow("未找到核心标记 —— 改用按类型三档处置"))
        screened, kept_b, removed_b, comp_rec = _screen_without_markers(text, req)
        nb, na, nr = len(kept_b) + len(removed_b), len(kept_b), len(removed_b)
        if kept_b or removed_b:
            _result_line(True, f"三档处置:模块 {_C.bold(str(nb))} → "
                         f"{_C.bold(str(na))} 个(删除 {nr}，压缩 {len(comp_rec)})")
        return _finish(screened, len(kept_b), removed_b, comp_rec)

    # ---- 路径 1:有核心标记 ----
    pre, rest  = text.split(_CORE_OPEN, 1)
    core, post = rest.split(_CORE_CLOSE, 1)
    core = core.strip()

    kept_pre,  kept_pre_b,  removed_pre,  comp_pre  = _screen_segment(pre,  req)
    kept_post, kept_post_b, removed_post, comp_post = _screen_segment(post, req)
    removed  = removed_pre + removed_post
    comp_rec = comp_pre + comp_post

    final = "\n\n".join(p for p in (kept_pre, core, kept_post) if p.strip())

    if len(final) > len(text):
        _result_line(False, _C.yellow(f"筛检后反而变长（{len(text)} → {len(final)} 字符）"
                                      f"—— 放弃筛检结果，返回 M3 输出"))
        return _finish(stripped_full, 0, [], [])   # 未实际筛检 → 模块统计记 0

    core_block = 1 if core.strip() else 0       # ← core 计 1 个固定保留模块
    n_kept    = len(kept_pre_b) + len(kept_post_b) + core_block
    n_removed = len(removed)
    n_before  = n_kept + n_removed

    lb, la = metrics["len_before"], len(final)
    saved  = lb - la
    _result_line(True, f"非核心三档处置完成：模块 {_C.bold(str(n_before))} → "
                 f"{_C.bold(str(n_kept))} 个（删除 {n_removed}，压缩 {len(comp_rec)}）；"
                 f"字符 {lb} → {la}（精简 {saved}，约 {saved * 100 // max(lb, 1)}%）")
    _detail(f"核心部分（M3 产出，原样保留，计为 1 个模块）：{len(core)} 字符")
    if removed:
        _detail(_C.yellow(f"删除 {len(removed)} 个非核心模块："))
        for b in removed:
            preview = (b.get("content") or "").replace("\n", " ")[:50]
            _detail(f"  {_C.red('✗')} [{b.get('content_type','other')}] {preview}…")
    if comp_rec:
        _detail(_C.yellow(f"压缩 {len(comp_rec)} 个非核心模块："))
        for r in comp_rec:
            lb_, la_ = len(r.get("before","")), len(r.get("after",""))
            _detail(f"  {_C.cyan('~')} [{r.get('content_type','other')}] {lb_} → {la_} 字符")
    if not removed and not comp_rec:
        _detail("非核心部分无可删除/压缩的模块")
    return _finish(final, n_kept, removed, comp_rec)


def _print_m5_substeps(m5: dict) -> None:
    if not m5 or m5.get("m5_skipped"):
        _result_line(False, _C.dim("M5 跳过（无工程师答案）"))
        return
    if m5.get("m5_error"):
        _result_line(False, _C.red(f"M5 评估失败：{m5['m5_error']}"))
        return

    cons = m5.get("consistency", {})
    # 工程师对照关闭时，cons 无 m3_core_vs_engineer，跳过一致性打印
    if "m3_core_vs_engineer" in cons:
        m3v = cons.get("m3_core_vs_engineer", {}).get("verdict", "?")
        m2v = cons.get("m2_core_vs_engineer", {}).get("verdict", "?")
        delta = cons.get("delta", "?")
        pruned = cons.get("m3_pruned_core", False)
        layer = _C.cyan("M3已剔除核心") if pruned else _C.dim("M3未动核心")
        _result_line(True, f"核心一致性[{layer}]：M3核心={_C.bold(m3v)} / M2核心={m2v} → {_C.cyan(delta)}")

    # 核心质量分（始终有）
    cq = m5.get("core_quality", {})
    if cq:
        m2q = (cq.get("m2_core") or {}).get("score")
        m3q = (cq.get("m3_core") or {}).get("score")
        if m2q is not None and m3q is not None:
            _result_line(True, f"核心质量：M2核心={m2q} → M3核心={m3q}（{cq.get('delta','?')}）")

    # M4 长度 + 简洁分（始终有）
    ml = m5.get("m4_length", {})
    if ml and ml.get("reduction_ratio") is not None:
        _result_line(True, f"M4缩减率：{ml['reduction_ratio']*100:.0f}%（去除 {ml.get('removed_chars',0)} 字符）")
    mc = m5.get("m4_concision", {})
    if mc and mc.get("scored"):
        ret = (mc.get("retention") or {}).get("score")
        con = (mc.get("concision") or {}).get("score")
        _result_line(True, f"M4简洁分：要点保持={ret} / 表达简洁={con}")

    dela = m5.get("deletion_analysis", {}).get("summary", {})
    if dela.get("total_deleted", 0):
        prec = dela.get("precision")
        prec_s = f"{prec * 100:.0f}%" if prec is not None else "—"
        _result_line(True, f"M4删除分析：{dela['true_negative']}正确删除 / "
                     f"{dela['false_negative']}误删，精度 {prec_s}")

    cov = m5.get("coverage", {}).get("summary", {})
    if cov.get("total_points"):
        r2 = cov.get("recall_m2_core")
        r3 = cov.get("recall_m3_core")
        r2_s = f"{r2 * 100:.0f}%" if r2 is not None else "—"
        r3_s = f"{r3 * 100:.0f}%" if r3 is not None else "—"
        _result_line(True, f"要点召回 M2核心→M3核心：{r2_s}→{r3_s}  "
                     f"补回{cov.get('points_recovered_by_m3',0)}/丢失{cov.get('points_lost_by_m3',0)}")


# =============================================================================
# Pipeline orchestrator
# =============================================================================

class VerilogPipeline:

    def __init__(self, cfg: Optional[PipelineConfig] = None):
        if cfg is not None:
            _apply_config(cfg)
        self.m1 = M1TaskClassifier()

    def run(self, title: str, question: str,
            dataset_label: dict | None = None,
            engineer_answer: str | None = None,
            baseline_answer: str | None = None,
            baseline_model: str | None = None) -> dict:
        dataset_label = dataset_label or {}
        result: dict  = {"title": title, "question": question[:500]}

        # ---- M1 ----
        _stage_banner("M1", "二次检查数据集已有分类")
        req = self.m1.classify(title, question, dataset_label)
        result["m1"] = req.to_dict()
        rev = req.classification_review or {}
        _result_line(True,
                     f"任务类型 = {_C.bold(req.category.primary.value)}"
                     + (f" (+{req.category.secondary.value})" if req.category.secondary else "")
                     + f"   [复检={rev.get('review','?')}，先验={rev.get('prior')}]")
        if rev.get("reason"):
            _substep(f"复检理由 : {rev['reason']}")
        _substep(f"验证目标 : {', '.join(v.value for v in req.verification)}")
        _substep(f"风险画像 : {req.risk_profile.to_dict()}")
        _substep(f"回答策略 : 长度={req.answer_policy.target_length}, "
                 f"代码={req.answer_policy.include_code}, "
                 f"testbench={'是' if req.answer_policy.include_testbench else '否'}, "
                 f"多选={'是' if req.answer_policy.list_multiple_options else '否'}")
        logger.info("[M1] review=%s prior=%s final=%s verify=%s",
                    rev.get("review"), rev.get("prior"), req.category.primary.value,
                    [v.value for v in req.verification])

        # ---- M2 ----
        _stage_banner("M2")
        draft = run_m2(req)
        result["m2_draft"] = draft
        _result_line(True, f"供下游使用的草稿：{len(draft)} 字符")

        # ---- M3 ----
        _stage_banner("M3", "宽松验证 — 仅精修核心答案部分")
        m3_output, feedback_log = run_m3(req, draft)
        result["m3_output"]   = m3_output
        result["m3_feedback"] = feedback_log

        # ---- M4 ----
        _stage_banner("M4", "筛检核心以外的所有部分（核心不变）")
        final_answer, m4_metrics = run_m4(m3_output, req)
        result["final_answer"] = final_answer
        result["m4_deleted_blocks"] = m4_metrics.get("deleted_blocks", [])
        result["m4_compressed_blocks"] = m4_metrics.get("compressed_blocks", [])

        # ---- 指标 ----
        m3_total  = len(feedback_log)
        m3_passed = sum(1 for r in feedback_log if r.get("passed"))
        result["metrics"] = {
            "category": req.category.primary.value,
            "m3_total": m3_total,
            "m3_passed": m3_passed,
            "m3_failed": m3_total - m3_passed,
            "m4_blocks_before": m4_metrics["blocks_before"],
            "m4_blocks_after": m4_metrics["blocks_after"],
            "m4_removed": m4_metrics["removed_count"],
            "m4_compressed": m4_metrics["compressed_count"],
            "m4_len_before": m4_metrics["len_before"],
            "m4_len_after": m4_metrics["len_after"],
            "m4_reduction": m4_metrics["reduction_ratio"],
        }

        # ---- M5 评估 ----
        if _cfg.enable_m5 and engineer_answer:
            from m5_evaluation import run_m5
            _stage_banner("M5", "对照工程师答案的 LLM-as-Judge 评估")
            m5 = run_m5(
                req             = req,
                draft           = draft,
                m3_output       = m3_output,
                final_answer    = final_answer,
                deleted_blocks  = m4_metrics.get("deleted_blocks", []),
                m3_feedback     = feedback_log,
                engineer_answer = engineer_answer,
                baseline_answer = baseline_answer,
                baseline_model  = baseline_model,
                judge_model_cfg = _cfg.models.judge,
                judge_temperature = _cfg.m5_judge_temperature,
                raw_llm_fn      = _raw_llm,
                strip_fences_fn = _strip_fences,
                enable_engineer_eval = _cfg.m5_enable_engineer_eval,   # ★ 开关接入
                compressed_blocks    = m4_metrics.get("compressed_blocks", []),  # ★ M4 压缩配对
            )
            result["m5"] = m5
            _print_m5_substeps(m5)
        elif _cfg.enable_m5:
            # 无工程师答案：仍可跑不依赖工程师答案的客观/绝对指标
            from m5_evaluation import run_m5
            _stage_banner("M5", "仅客观/绝对指标（无工程师答案）")
            m5 = run_m5(
                req             = req,
                draft           = draft,
                m3_output       = m3_output,
                final_answer    = final_answer,
                deleted_blocks  = m4_metrics.get("deleted_blocks", []),
                m3_feedback     = feedback_log,
                engineer_answer = None,
                judge_model_cfg = _cfg.models.judge,
                judge_temperature = _cfg.m5_judge_temperature,
                raw_llm_fn      = _raw_llm,
                strip_fences_fn = _strip_fences,
                enable_engineer_eval = False,
                compressed_blocks    = m4_metrics.get("compressed_blocks", []),
            )
            result["m5"] = m5
            _print_m5_substeps(m5)

        return result


# =============================================================================
# 指标聚合 & 批处理（并发 + 进度条 + 自动保存）
# =============================================================================

class MetricsAggregator:
    _FIELDS = ("records", "m3_total", "m3_passed", "m3_failed",
               "blocks_before", "blocks_after", "m4_removed", "m4_compressed",
               "len_before", "len_after")

    def __init__(self):
        self._lock   = threading.Lock()
        self._by_cat: dict[str, dict] = {}

    def add(self, m: dict) -> None:
        with self._lock:
            d = self._by_cat.setdefault(m["category"], {k: 0 for k in self._FIELDS})
            d["records"]    += 1
            d["m3_total"]   += m.get("m3_total", 0)
            d["m3_passed"]  += m.get("m3_passed", 0)
            d["m3_failed"]  += m.get("m3_failed", 0)
            d["m4_removed"] += m.get("m4_removed", 0)
            d["m4_compressed"] += m.get("m4_compressed", 0)
            d["len_before"] += m.get("m4_len_before", 0)
            d["len_after"]  += m.get("m4_len_after", 0)
            d["blocks_before"] += m.get("m4_blocks_before", 0)
            d["blocks_after"] += m.get("m4_blocks_after", 0)

    @staticmethod
    def _row(name, d, cols, bold=False):
        tot = d["m3_total"]
        pas = d["m3_passed"]
        recs = d["records"]
        rate = f"{pas * 100 / tot:.1f}%" if tot else "—"
        bb = f"{d['blocks_before'] / recs:.1f}" if recs else "—"
        ba = f"{d['blocks_after'] / recs:.1f}" if recs else "—"
        avg = f"{d['m4_removed'] / recs:.2f}" if recs else "—"
        avgc = f"{d['m4_compressed'] / recs:.2f}" if recs else "—"
        lb = d["len_before"]
        red = f"{(lb - d['len_after']) * 100 / lb:.1f}%" if lb else "—"
        cells = [name, recs, f"{pas}/{tot}", rate, d["m3_failed"],
                 f"{bb}→{ba}", avg, avgc, red]
        row = "".join(_pad(c, w) for c, (_, w) in zip(cells, cols))
        return _C.bold(row) if bold else row

    def render(self) -> None:
        with self._lock:
            cats = {k: dict(v) for k, v in self._by_cat.items()}
        if not cats:
            print("\n（无可统计的成功记录）\n")
            return

        cols = [("类别", 16), ("样本", 6), ("M3通过/总", 14), ("通过率", 9),
                ("未通过", 8), ("M4模块均原→现", 16), ("M4均删", 8), ("M4均压", 8),
                ("长度缩减", 10)]
        line_w = sum(w for _, w in cols)

        print("\n" + "=" * line_w)
        print(_C.bold("  指标统计（分类别）"))
        print("=" * line_w)
        print("".join(_pad(h, w) for h, w in cols))
        print("-" * line_w)

        total = {k: 0 for k in self._FIELDS}
        for cat in sorted(cats, key=lambda c: cats[c]["records"], reverse=True):
            d = cats[cat]
            for k in self._FIELDS:
                total[k] += d[k]
            print(self._row(cat, d, cols))

        print("-" * line_w)
        print(self._row("总计", total, cols, bold=True))
        print("=" * line_w)
        print(_C.dim("说明：M3通过率 = 辩论通过数 / 总辩论数；未通过 = 失败的辩论数；"))
        print(_C.dim("      M4均删/均压 = 平均删除/压缩的非核心模块数；长度缩减 = M4 筛检导致的字符数下降比例。"))
        print()


def _metric_line(done: int, total: int, idx: int, res: dict) -> str:
    head = _C.dim(f"[{done}/{total}]") + f" #{idx + 1}"
    if res.get("skipped"):
        return f"{_C.dim('·')} {head} 跳过（空问题）"
    if res.get("error"):
        return f"{_C.red('✗')} {head} 错误: {str(res['error'])[:80]}"
    m    = res.get("metrics") or {}
    cat  = m.get("category", "?")
    tot  = m.get("m3_total", 0)
    pas  = m.get("m3_passed", 0)
    fail = m.get("m3_failed", 0)
    rate = (pas * 100 / tot) if tot else 0.0
    nb = m.get("m4_blocks_before", 0)
    na = m.get("m4_blocks_after", 0)
    rmv = m.get("m4_removed", 0)
    cmp_ = m.get("m4_compressed", 0)
    lb = m.get("m4_len_before", 0)
    la = m.get("m4_len_after", 0)
    red = m.get("m4_reduction", 0.0) * 100
    return (f"{_C.green('✔')} {head} [{_C.bold(cat)}] "
            f"M3 {pas}/{tot}({rate:.0f}%) 未通过{fail} | "
            f"M4 模块 {nb}→{na}(删{rmv}/压{cmp_}) | 长度 {lb}→{la} (-{red:.1f}%)")


def _auto_save(results: list, output_path: str, lock: threading.Lock) -> None:
    """线程安全地将当前已完成的结果保存到磁盘（中间结果）。"""
    with lock:
        valid = [r for r in results if r is not None]
    if not valid:
        return
    tmp_path = output_path + ".tmp"
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(valid, f, ensure_ascii=False, indent=2)
        if os.path.exists(output_path):
            os.replace(tmp_path, output_path)
        else:
            os.rename(tmp_path, output_path)
        logger.info("Auto-saved %d results to %s", len(valid), output_path)
    except Exception as exc:
        logger.warning("Auto-save failed: %s", exc)


# =============================================================================
# 实验归档：每次跑批新建 run_<时间戳> 子文件夹，保存结果 + 模型/参数元数据
# =============================================================================

def _safe_model_cfg(mc) -> dict:
    """提取单个模型配置，绝不包含 api_key。"""
    if mc is None:
        return {}
    return {
        "model":    getattr(mc, "model", None),
        "base_url": getattr(mc, "base_url", None),
        # 故意不输出 api_key
    }


def _collect_run_metadata(cfg: "PipelineConfig", total_records: int) -> dict:
    """汇总本次实验的模型信息 + 主要参数（不含任何密钥）。"""
    import platform
    m = cfg.models
    return {
        "run_time":     time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_path":   cfg.input_path,
        "dataset_size": total_records,
        "models": {
            "default":     _safe_model_cfg(m.default),
            "readability": _safe_model_cfg(m.readability),
            "consistency": _safe_model_cfg(m.consistency),
            "concise":     _safe_model_cfg(m.concise),
            "codegen":     _safe_model_cfg(m.codegen),
            "judge":       _safe_model_cfg(m.judge),
        },
        "parameters": {
            "max_verify_rounds":  cfg.max_verify_rounds,
            "max_autogen_turns":  cfg.max_autogen_turns,
            "llm_temperature":    cfg.llm_temperature,
            "batch_max_items":    cfg.batch_max_items,
            "batch_delay_sec":    cfg.batch_delay_sec,
            "concurrency":        cfg.concurrency,
            "auto_save_interval": cfg.auto_save_interval,
            "ablation_mode": cfg.ablation_mode,
        },
        "m5": {
            "enable_m5":                cfg.enable_m5,
            "m5_accepted_answer_field": cfg.m5_accepted_answer_field,
            "m5_judge_temperature":     cfg.m5_judge_temperature,
            "m5_baseline_model":        cfg.m5_baseline_model,
            "m5_enable_engineer_eval":  cfg.m5_enable_engineer_eval,
        },
        "environment": {
            "python":   platform.python_version(),
            "platform": platform.platform(),
        },
    }


def _setup_run_dir(cfg: "PipelineConfig") -> tuple[str, str]:
    """在 output_path 所在目录下新建 run_<时间戳> 子文件夹。
    返回 (run_dir, results_path)，并把 cfg.output_path 重定向到子文件夹内。"""
    out_dir = os.path.dirname(os.path.abspath(cfg.output_path)) or "."
    base    = os.path.basename(cfg.output_path) or "pipeline_output.json"
    stamp   = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_dir, stamp)
    os.makedirs(run_dir, exist_ok=True)
    results_path = os.path.join(run_dir, base)
    return run_dir, results_path


def _write_run_metadata(run_dir: str, meta: dict) -> None:
    """写机器可读 run_meta.json + 人类可读 run_info.txt。"""
    with open(os.path.join(run_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    lines = ["=" * 60, "实验运行信息 (Run Info)", "=" * 60,
             f"运行时间   : {meta['run_time']}",
             f"输入数据   : {meta['input_path']}",
             f"数据规模   : {meta['dataset_size']} 条", "",
             "模型配置 (按角色):"]
    for role, mc in meta["models"].items():
        lines.append(f"  {role:<12}: model={mc.get('model')}  base_url={mc.get('base_url')}")
    lines += ["", "主要参数:"]
    for k, v in meta["parameters"].items():
        lines.append(f"  {k:<20}: {v}")
    lines += ["", "M5 评估:"]
    for k, v in meta["m5"].items():
        lines.append(f"  {k:<26}: {v}")
    lines += ["", "运行环境:",
              f"  python  : {meta['environment']['python']}",
              f"  platform: {meta['environment']['platform']}",
              "=" * 60]
    with open(os.path.join(run_dir, "run_info.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def process_dataset(cfg: PipelineConfig) -> list[dict]:
    """处理整个 JSON 数据集，带进度条、自动保存、分类别汇总。"""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    _apply_config(cfg)
    _set_verbose(cfg.concurrency <= 1)

    with open(cfg.input_path, encoding="utf-8") as f:
        records = json.load(f)
    if cfg.batch_max_items > 0:
        records = records[:cfg.batch_max_items]

    total = len(records)

    # ---- 新建本次实验子文件夹 + 写元数据 ----
    run_dir, results_path = _setup_run_dir(cfg)
    cfg.output_path = results_path  # 所有结果/中间保存都落到子文件夹
    meta = _collect_run_metadata(cfg, total)
    _write_run_metadata(run_dir, meta)
    print()
    print(_C.bold(f"  📁 本次实验目录: {run_dir}"))
    print(_C.dim(f"     结果将保存到: {results_path}"))
    print(_C.dim(f"     模型/参数元数据: run_meta.json / run_info.txt"))

    pipeline = VerilogPipeline(cfg)
    agg = MetricsAggregator()
    from m5_evaluation import M5Aggregator
    m5_agg = M5Aggregator(ablation_mode=cfg.ablation_mode)
    results: list[Optional[dict]] = [None] * total
    results_lock = threading.Lock()
    progress     = ProgressTracker(total)
    logger.info("Starting batch — %d records, concurrency=%d, auto_save_interval=%d",
                total, cfg.concurrency, cfg.auto_save_interval)

    print()
    print(_C.bold(f"  处理 {total} 条记录  |  并发={cfg.concurrency}  |  "
                  f"每 {cfg.auto_save_interval} 条自动保存"))
    print()

    def _work(idx: int, rec: dict) -> tuple[int, dict]:
        from m5_evaluation import extract_engineer_answer, extract_baseline_llm_answer
        title    = rec.get("title", "")
        question = rec.get("question", "")
        if not str(question).strip():
            return idx, {"title": title, "skipped": True}
        engineer_answer = None
        baseline_answer = baseline_model = None
        if cfg.enable_m5:
            engineer_answer = extract_engineer_answer(
                rec, accepted_field=cfg.m5_accepted_answer_field)
            baseline_answer, baseline_model = extract_baseline_llm_answer(
                rec, prefer_model=(cfg.m5_baseline_model or None))
        try:
            if _VERBOSE:
                _safe_print("")
                _safe_print(_C.blue("█" * 65))
                _safe_print(_C.blue("█") + _C.bold(f"  [{idx + 1}/{total}] {title[:55]}"))
                _safe_print(_C.blue("█" * 65))
            res = pipeline.run(title, question, rec.get("classification", {}),
                               engineer_answer=engineer_answer,
                               baseline_answer=baseline_answer,
                               baseline_model=baseline_model)
        except Exception as exc:
            logger.error("[%d/%d] Processing failed: %s", idx + 1, total, exc)
            res = {"title": title, "error": str(exc)}
        return idx, res

    done_count = 0

    if cfg.concurrency <= 1:
        # 串行
        for idx, rec in enumerate(records):
            _, res = _work(idx, rec)
            with results_lock:
                results[idx] = res
            done_count += 1
            has_metrics = res.get("metrics") is not None
            if has_metrics:
                agg.add(res["metrics"])
            _m5 = res.get("m5")
            if _m5 and not _m5.get("m5_skipped") and not _m5.get("m5_error"):
                m5_agg.add(res.get("metrics", {}).get("category", "unknown"), _m5)
            progress.update(ok=has_metrics and not res.get("error"),
                            skip=res.get("skipped", False))
            _safe_print(_metric_line(done_count, total, idx, res))
            progress.print_bar()

            if cfg.auto_save_interval > 0 and done_count % cfg.auto_save_interval == 0:
                _auto_save(results, cfg.output_path, results_lock)
                _safe_print(_C.dim(f"  💾 自动保存 ({done_count}/{total})"))

            if cfg.batch_delay_sec > 0:
                time.sleep(cfg.batch_delay_sec)
    else:
        # 并发
        with ThreadPoolExecutor(max_workers=cfg.concurrency) as ex:
            futures = [ex.submit(_work, idx, rec) for idx, rec in enumerate(records)]
            for fut in as_completed(futures):
                idx, res = fut.result()
                with results_lock:
                    results[idx] = res
                done_count += 1
                has_metrics = res.get("metrics") is not None
                if has_metrics:
                    agg.add(res["metrics"])
                _m5 = res.get("m5")
                if _m5 and not _m5.get("m5_skipped") and not _m5.get("m5_error"):
                    m5_agg.add(res.get("metrics", {}).get("category", "unknown"), _m5)
                progress.update(ok=has_metrics and not res.get("error"),
                                skip=res.get("skipped", False))
                _safe_print(_metric_line(done_count, total, idx, res))
                progress.print_bar()

                if cfg.auto_save_interval > 0 and done_count % cfg.auto_save_interval == 0:
                    _auto_save(results, cfg.output_path, results_lock)
                    _safe_print(_C.dim(f"  💾 自动保存 ({done_count}/{total})"))

    progress.finish()

    final_results = [r for r in results if r is not None]
    os.makedirs(os.path.dirname(os.path.abspath(cfg.output_path)), exist_ok=True)
    with open(cfg.output_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)

    print()
    print(_C.green(_C.bold(f"  ✔ 最终结果已保存: {cfg.output_path}")))
    print(_C.dim(f"    共 {len(final_results)} 条有效记录"))
    print()

    logger.info("Done! Results written to: %s", cfg.output_path)
    _print_summary(final_results)
    agg.render()
    m5_agg.render(_C, _pad)
    # 把本次 run 的关键统计也存一份到子文件夹（便于回溯）
    try:
        summary_path = os.path.join(run_dir, "run_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Run: {run_dir}\n")
            f.write(f"Records: {len(final_results)}\n")
            f.write(f"Models: {json.dumps(meta['models'], ensure_ascii=False)}\n")
            f.write(f"Params: {json.dumps(meta['parameters'], ensure_ascii=False)}\n")
        logger.info("Run summary written to %s", summary_path)
    except Exception as exc:
        logger.warning("write run_summary failed: %s", exc)
    return final_results


def _print_summary(results: list[dict]) -> None:
    from collections import Counter
    cats = Counter(
        r.get("m1", {}).get("category", {}).get("primary", "unknown")
        for r in results if r.get("metrics")
    )
    print("\n" + "=" * 55)
    print(f"Pipeline Summary ({len(results)} records processed)")
    print("=" * 55)
    for cat, cnt in cats.most_common():
        bar = "#" * (cnt * 20 // max(cats.values())) if cats else ""
        print(f"  {cat:<25} {cnt:>4}  {bar}")
    print("=" * 55 + "\n")


TEST_CASES = [
    (
        "Vivado 2016.1: Upon synthesis it is removing useful logic. Verilog",
        "I am currently working on building a soft core processor. "
        "Even though simulation works perfectly, synthesis throws warnings: "
        "[Synth 8-3332] Sequential element (FM1/out_reg[31]) is unused and will be removed.",
        {"category": "Debugging", "subcategory": "synthesis-error",
         "reason": "User reports a synthesis warning about logic being removed and asks why."},
    ),
    (
        "How is the program block controlling the clock output in this code?",
        "This is a simple SV program. I don't know how the line repeat(4) @ar.cb "
        "keeps controlling the entire clock. If I comment out that line, the clock stops.",
        {"category": "Concept", "subcategory": "simulation-semantics",
         "reason": "User wants to understand how a language construct controls clocking."},
    ),
    (
        "Generate a parameterized synchronous FIFO in Verilog",
        "Please implement a parameterizable synchronous FIFO with configurable depth and width. "
        "It should support full/empty flags and be synthesizable on FPGA.",
        {},
    ),
    (
        "Reduce critical path in this 32-bit adder",
        "My ripple-carry based 32-bit adder fails timing at 200 MHz. "
        "How can I restructure it to meet timing without changing the interface?",
        {"category": "Optimization", "subcategory": "timing",
         "reason": "User wants to restructure working code to meet a timing target."},
    ),
]


def _run_single_tests(cfg: PipelineConfig) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)s  %(message)s")
    _apply_config(cfg)
    _set_verbose(True)

    pipeline = VerilogPipeline(cfg)
    print("=" * 65)
    print("Verilog Multi-Agent Pipeline — Single-Item Test (宽松模式)")
    print("=" * 65)

    for i, (title, question, ds_label) in enumerate(TEST_CASES, 1):
        print()
        print(_C.blue("█" * 65))
        print(_C.blue("█") + _C.bold(f"  [Case {i}] {title[:55]}"))
        print(_C.blue("█" * 65))
        result = pipeline.run(title, question, ds_label)

        m1 = result.get("m1", {})
        print()
        print(_C.bold("  ── 本条结果摘要 ──"))
        print(f"  M1 type         : {m1.get('category', {}).get('primary', '-')}")
        print(f"  M1 review       : {m1.get('classification_review', {})}")
        print(f"  M1 verification : {m1.get('verification', [])}")
        print(f"  M1 risk         : {m1.get('risk_profile', {})}")
        print(f"  M1 policy       : {m1.get('answer_policy', {})}")
        fb_log = result.get("m3_feedback", [])
        passed = sum(1 for r in fb_log if r.get("passed"))
        print(f"  M3 core items   : {passed}/{len(fb_log)} passed")
        for rec in fb_log:
            status = "PASS" if rec.get("passed") else "FAIL"
            print(f"    [{status}] {rec.get('item_id','?')} — {rec.get('title','')}: "
                  f"{rec.get('feedback','')[:120]}")
        m4m = result.get("metrics", {})
        print(f"  M4 metrics      : removed={m4m.get('m4_removed','-')} / "
              f"compressed={m4m.get('m4_compressed','-')} blocks | "
              f"len {m4m.get('m4_len_before','-')}→{m4m.get('m4_len_after','-')} "
              f"(-{m4m.get('m4_reduction', 0) * 100:.1f}%)")
        print(f"\n  [Final Answer]\n{result.get('final_answer', '[none]')}")

    print("\n" + "=" * 65)
    print("Tests complete.")
    print("=" * 65)


def main() -> None:
    api_key = os.environ.get("LLM_API_KEY", "")
    if not api_key:
        if not api_key:
            sys.exit(1)

    default_model = ModelConfig(
        api_key  = api_key,
        base_url = "",
        model    = "",
    )
    judge = ModelConfig(
        api_key  = api_key,
        base_url = "",
        model    = "",
    )



    model_routes = ModelRouteConfig(
        default=default_model,
        readability=default_model,  # M2 conceptual
        consistency=default_model,  # M2 debugging
        concise=default_model,  # M2 optimization
        codegen=default_model,  # M2 generation
        m3=default_model,  # ← M3 验证/辩论,可换更强模型避免"弱模型审强模型"
        m3_split=default_model,
        m4=default_model,  # ← M4 筛检,纯结构化任务,可换便宜模型省成本
        judge=judge,  # M5 评估
    )

    input_path  = r""
    output_path = r""

    max_verify_rounds = 1
    max_autogen_turns = 3
    llm_temperature   = 0.0   # 确定性：消除重跑间的随机波动，保证结论可复现

    batch_max_items    = 0   # 只跑前 100 条；改回 0 = 全部
    batch_delay_sec    = 0.0
    concurrency        = 64
    auto_save_interval = 5

    cfg = PipelineConfig(
        input_path         = input_path,
        output_path        = output_path,
        models             = model_routes,
        max_verify_rounds  = max_verify_rounds,
        max_autogen_turns  = max_autogen_turns,
        llm_temperature    = llm_temperature,
        batch_max_items    = batch_max_items,
        batch_delay_sec    = batch_delay_sec,
        concurrency        = concurrency,
        auto_save_interval = auto_save_interval,
        enable_m5                = True,
        m5_accepted_answer_field = "accepted_answer",
        m5_judge_temperature     = 0.0,
        m5_baseline_model        = "",
        m5_enable_engineer_eval  = False,   # ★ 工程师对照默认关；要完整一致性指标时改 True

        ablation_mode = "full",
    )

    if "--input"       in sys.argv: cfg.input_path        = sys.argv[sys.argv.index("--input") + 1]
    if "--output"      in sys.argv: cfg.output_path       = sys.argv[sys.argv.index("--output") + 1]
    if "--n"           in sys.argv: cfg.batch_max_items   = int(sys.argv[sys.argv.index("--n") + 1])
    if "--concurrency" in sys.argv: cfg.concurrency       = int(sys.argv[sys.argv.index("--concurrency") + 1])
    if "--save-every"  in sys.argv: cfg.auto_save_interval = int(sys.argv[sys.argv.index("--save-every") + 1])
    if "--no-m5"          in sys.argv: cfg.enable_m5 = False
    if "--engineer-eval"  in sys.argv: cfg.m5_enable_engineer_eval = True
    if "--accepted-field" in sys.argv: cfg.m5_accepted_answer_field = sys.argv[sys.argv.index("--accepted-field") + 1]
    if "--baseline-model" in sys.argv: cfg.m5_baseline_model = sys.argv[sys.argv.index("--baseline-model") + 1]
    if "--ablation" in sys.argv:
        cfg.ablation_mode = sys.argv[sys.argv.index("--ablation") + 1]

    if "--test" in sys.argv:
        _run_single_tests(cfg)
    else:
        process_dataset(cfg)


if __name__ == "__main__":
    main()