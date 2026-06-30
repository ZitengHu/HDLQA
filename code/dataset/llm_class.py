
import json
import os
import re
import sys
import time
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Tuple

import httpx
from openai import OpenAI
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


# ─────────────────────────────────────────────────────────────
# 1. API
# ─────────────────────────────────────────────────────────────

def call_llm(query: str, api_key: str, base_url: str,
             model_name: str, temperature: float = 0.2) -> str:
    """Single LLM call. Raises on HTTP / API errors."""
    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=httpx.Client(timeout=60.0),
    )
    resp = client.chat.completions.create(
        messages=[{"role": "user", "content": query}],
        model=model_name,
        temperature=temperature,
    )
    return resp.choices[0].message.content


# ─────────────────────────────────────────────────────────────
# 2. Classification prompt
# ─────────────────────────────────────────────────────────────

CLASSIFICATION_PROMPT = """\
You are an expert hardware-design Q&A classifier. Your task is to assign exactly one \
category and subcategory to the following Stack Overflow post about Verilog or hardware design.

=== INPUT DATA ===
ID    : {data_id}
Title : {title}
Body  : {body}
Answers : {answers}

=== TAXONOMY ===
1. OPTIMIZATION — Primary intent: IMPROVE an existing (working) design on PPAC, readability, or security / reliability axes.
   The design functions correctly but needs to be better. Code is NOT required — a question about HOW TO improve performance, power, area, cost, security, reliability, or CODE STYLE of a design is ALWAYS Optimization.
   • ppa-optimization       : timing (critical path), power, area, or cost improvement
   • security-optimization  : race conditions, uninitialized states, glitches, sim-synth mismatch, metastability, side-channel leakage, testability gaps ... Includes questions of the form "does X guarantee Y reliability/safety property" — even if phrased as a conceptual question, the underlying concern is a security/reliability design goal → always Optimization.
   • readability-optimization : code style, naming conventions, modularity, comments, verbosity reduction, or general "clean code" refactoring where functionality remains the same but maintainability improves.

2. GENERATION — Primary intent: BUILD new hardware. No complete implementation exists yet; the user describes WHAT a circuit should DO (behavior, interface, or algorithm) and wants a working implementation created.
   Key signal: look for functional requirements ("I need a module that...", "implement a ... which...", "how do I write ... to achieve ..."). A partial stub or short example snippet does NOT disqualify Generation — classify as Generation if a substantial working implementation is still absent.
   • logic-generation       : FSMs, counters, memories, FIFOs, arbiters, processors, comm controllers ...
   • arithmetic-generation  : adders, multipliers, dividers, MACs, FP units, DSP blocks ...

3. DEBUGGING — Primary intent: FIX an explicit error or unexpected behavior in code that already exists and is substantially complete. The post must contain (a) existing code AND (b) a described defect or wrong output.
   • syntax-error           : compilation / elaboration error (undeclared id, port mismatch, missing token ...)
   • functional-error       : the design compiles and elaborates without errors but SIMULATION gives wrong results (wrong waveform, wrong FSM state, wrong counter value ...)
   • synthesis-error        : tool-reported synthesis / implementation error or WARNING during synthesis/P&R (unintended latch, unresolved signal, timing violation flagged by tool ...)

4. CONCEPTUAL — Primary intent: UNDERSTAND a language feature, syntax rule, or EDA tool. Code presence is irrelevant; what matters is the user is asking "how does X work" or "what does Y mean," not trying to fix a bug, build new hardware, or improve a design. A question qualifies as Conceptual ONLY IF it involves no PPAC, security, reliability, or readability concern — if any such concern is present, classify as Optimization instead.
   • syntax-explanation     : language features, semantics, idioms (blocking vs non-blocking, sensitivity lists, generate, parameter vs localparam ...). Line-level or construct-level questions about what existing code MEANS or HOW a language construct works — NOT about building new functionality.
   • tool-usage             : compiler / EDA tool operation (Vivado options, ModelSim commands, VCS flags, Quartus IP config ...)

=== BOUNDARY DISAMBIGUATION ===
The following pairs are frequently confused. Apply these tie-breaking rules:
│ Confused pair              │ Deciding question                                                                                      │
│----------------------------│--------------------------------------------------------------------------------------------------------│
│ Conceptual vs Optimization │ Is the user asking how to IMPROVE a design's PPAC, readability, security, or reliability — even without code? → Optimization. Does the question involve glitch-free, metastability, race conditions, or other reliability properties, even if phrased as "does X guarantee Y"? → Optimization (security-optimization). Is the user asking what a language feature / syntax / tool option MEANS or HOW IT WORKS, with no reliability concern involved? → Conceptual. │
│ Generation vs Conceptual   │ Does the user state a functional requirement and ask for an implementation, even if a short stub exists? → Generation. Is the user asking what a specific construct or line of code MEANS, with no new circuit to build? → Conceptual (syntax-explanation). │
│ Conceptual vs Debugging    │ Is there a concrete defect to fix? → Debugging. Is the user asking what something means even with code present? → Conceptual. │
│ Debugging vs Optimization  │ Is the design broken (wrong output / error)? → Debugging. Is it correct but slow / large / unreadable / insecure? → Optimization. │
│ Debugging vs Generation    │ Does substantial code already exist? → Debugging. Is the user starting from scratch or a stub? → Generation. │
│ functional-error vs        │ Did the error appear during SIMULATION (wrong waveform/output at runtime)? → functional-error.          │
│ synthesis-error            │ Did the error/warning appear during SYNTHESIS or P&R tool execution? → synthesis-error.                 │
│ security-optimization vs   │ Is the vulnerability a design-level concern (race, leakage)? → security-optimization.                   │
│ synthesis-error            │ Is it a tool-reported warning with no design-level fix needed? → synthesis-error.                       │

=== OUTPUT FORMAT ===
Return ONLY a valid JSON object. No markdown fences, no extra text.
{{
  "id": "{data_id}",
  "category": "Conceptual|Debugging|Generation|Optimization",
  "subcategory": "syntax-explanation|tool-usage|syntax-error|functional-error|synthesis-error|logic-generation|arithmetic-generation|ppa-optimization|security-optimization|readability-optimization",
  "reason": "<one sentence citing specific evidence from the post>"
}}
"""

VALID_CATEGORIES = {"Conceptual", "Debugging", "Generation", "Optimization"}

VALID_SUBCATEGORIES = {
    # Conceptual
    "syntax-explanation",
    "tool-usage",
    # Debugging
    "syntax-error",
    "functional-error",
    "synthesis-error",
    # Generation
    "logic-generation",
    "arithmetic-generation",
    # Optimization
    "ppa-optimization",
    "security-optimization",
    "readability-optimization",
}

# Maximum words sent to LLM from question body (avoids token bloat)
MAX_BODY_WORDS = 4000000


# ─────────────────────────────────────────────────────────────
# 3. Ground truth loading
# ─────────────────────────────────────────────────────────────

def load_ground_truth(excel_path: str) -> List[Tuple[str, str]]:
    """
    Load ground truth from an Excel file with NO header row.
    Column A = category (e.g. "Conceptual"), Column B = subcategory (e.g. "tool-usage").
    Returns a list of (category, subcategory) tuples in file order.
    """
    try:
        import openpyxl
    except ImportError:
        raise ImportError("openpyxl is required to read Excel files. Run: pip install openpyxl")

    wb = openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
    ws = wb.active
    ground_truth = []
    for row in ws.iter_rows(values_only=True):
        cat = str(row[0]).strip() if row[0] is not None else ""
        sub = str(row[1]).strip() if row[1] is not None else ""
        if cat:  # skip completely empty rows
            ground_truth.append((cat, sub))
    wb.close()
    print(f"  Ground truth loaded: {len(ground_truth):,} items from {excel_path}")
    return ground_truth


# ─────────────────────────────────────────────────────────────
# 4. Data loading
# ─────────────────────────────────────────────────────────────

def load_data(json_file_path: str) -> List[Dict]:
    with open(json_file_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            f.seek(0)
            data = []
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))

    if isinstance(data, dict) and "posts" in data:
        return data["posts"]
    if isinstance(data, list):
        return data
    raise ValueError("JSON must be a list or a dict with a 'posts' key.")


# ─────────────────────────────────────────────────────────────
# 5. Checkpoint helpers
# ─────────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_file: str) -> Dict[str, Dict]:
    """Return {question_id: item} for already-processed items."""
    done: Dict[str, Dict] = {}
    if not os.path.exists(checkpoint_file):
        return done
    with open(checkpoint_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                qid = str(rec.get("question_id", ""))
                if qid:
                    done[qid] = rec
            except json.JSONDecodeError:
                pass
    return done


def append_checkpoint(checkpoint_file: str, record: Dict) -> None:
    """Append one completed record to the checkpoint JSONL."""
    with open(checkpoint_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────
# 6. Single-item classification
# ─────────────────────────────────────────────────────────────

def _parse_response(response: str, data_id: str) -> Dict:
    """
    Extract a valid JSON classification dict from raw LLM output.
    Returns category='parse_error' on failure — never silently falls
    back to a real category.
    """
    cleaned = re.sub(r"```(?:json)?", "", response, flags=re.IGNORECASE)
    cleaned = cleaned.strip().strip("`").strip()

    for candidate in (cleaned, response):
        try:
            obj = json.loads(candidate)
            if obj.get("category") in VALID_CATEGORIES:
                if obj.get("subcategory") not in VALID_SUBCATEGORIES:
                    obj["subcategory"] = "general"
                return obj
        except (json.JSONDecodeError, AttributeError):
            pass

    m = re.search(r"\{.*?\}", response, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group())
            if obj.get("category") in VALID_CATEGORIES:
                if obj.get("subcategory") not in VALID_SUBCATEGORIES:
                    obj["subcategory"] = "general"
                return obj
        except (json.JSONDecodeError, AttributeError):
            pass

    return {
        "id":          data_id,
        "category":    "parse_error",
        "subcategory": "parse_error",
        "reason":      f"Could not parse LLM response: {response[:200]}",
    }


def classify_one(item: Dict, api_key: str, base_url: str,
                 model_name: str, temperature: float,
                 max_retries: int = 5) -> Dict:

    title   = item.get("title", "")
    body    = item.get("question", "")
    data_id = str(item.get("question_id", "unknown"))

    answers = item.get("answers", [])
    answers_parts = []
    for i, ans in enumerate(answers, 1):
        ans_body = ans.get("body", ans.get("content", ""))
        ans_type = ans.get("type", "")
        answers_parts.append(f"[Answer {i} | {ans_type}]\n{ans_body}")
    answers_text = "\n\n".join(answers_parts) if answers_parts else "N/A"

    query = CLASSIFICATION_PROMPT.format(
        data_id=data_id,
        title=title,
        body=body,
        answers=answers_text,
    )

    last_error: Optional[str] = None
    for attempt in range(max_retries):
        try:
            response = call_llm(query, api_key, base_url, model_name, temperature)
            item["classification"] = _parse_response(response, data_id)
            return item
        except Exception as exc:
            last_error = str(exc)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)

    item["classification"] = {
        "id":          data_id,
        "category":    "api_error",
        "subcategory": "api_error",
        "reason":      f"API error after {max_retries} attempts: {last_error}",
    }
    return item


# ─────────────────────────────────────────────────────────────
# 7. Accuracy tracker
# ─────────────────────────────────────────────────────────────

class AccuracyTracker:
    """
    Tracks real-time category-level and subcategory-level accuracy
    against a ground truth list aligned by position.
    """

    def __init__(self, ground_truth: List[Tuple[str, str]]):
        # ground_truth[i] = (gt_category, gt_subcategory) for the i-th data item
        self.ground_truth = ground_truth

        # Running counters (only items that have a GT entry and a valid prediction)
        self.n_compared     = 0   # items with GT entries evaluated so far
        self.cat_correct    = 0   # category matches
        self.sub_correct    = 0   # subcategory matches (exact both)
        self.error_count    = 0   # parse_error / api_error — excluded from accuracy

        # Per-category breakdown: {gt_cat: {"total": N, "cat_hit": N, "sub_hit": N}}
        self.per_cat: Dict[str, Dict[str, int]] = {}

    def update(self, global_idx: int, predicted_cat: str, predicted_sub: str) -> Optional[Dict]:
        """
        Compare prediction at global_idx against ground truth.
        Returns a result dict, or None if no GT entry exists for this index.
        """
        if global_idx >= len(self.ground_truth):
            return None  # no GT for this item

        gt_cat, gt_sub = self.ground_truth[global_idx]

        # Skip error predictions from accuracy computation
        is_error = predicted_cat in ("parse_error", "api_error")
        if is_error:
            self.error_count += 1
            return {
                "gt_cat": gt_cat, "gt_sub": gt_sub,
                "pred_cat": predicted_cat, "pred_sub": predicted_sub,
                "cat_match": False, "sub_match": False,
                "excluded": True,
            }

        self.n_compared += 1

        cat_match = (predicted_cat == gt_cat)
        sub_match = (predicted_cat == gt_cat and predicted_sub == gt_sub)

        if cat_match:
            self.cat_correct += 1
        if sub_match:
            self.sub_correct += 1

        # Per-category stats
        if gt_cat not in self.per_cat:
            self.per_cat[gt_cat] = {"total": 0, "cat_hit": 0, "sub_hit": 0}
        self.per_cat[gt_cat]["total"]   += 1
        self.per_cat[gt_cat]["cat_hit"] += int(cat_match)
        self.per_cat[gt_cat]["sub_hit"] += int(sub_match)

        return {
            "gt_cat": gt_cat, "gt_sub": gt_sub,
            "pred_cat": predicted_cat, "pred_sub": predicted_sub,
            "cat_match": cat_match, "sub_match": sub_match,
            "excluded": False,
        }

    @property
    def cat_accuracy(self) -> float:
        return self.cat_correct / self.n_compared * 100 if self.n_compared else 0.0

    @property
    def sub_accuracy(self) -> float:
        return self.sub_correct / self.n_compared * 100 if self.n_compared else 0.0

    def summary(self) -> Dict:
        return {
            "n_compared":           self.n_compared,
            "error_excluded":       self.error_count,
            "category_accuracy":    round(self.cat_accuracy, 2),
            "subcategory_accuracy": round(self.sub_accuracy, 2),
            "per_category": {
                cat: {
                    "total":               v["total"],
                    "category_acc_pct":    round(v["cat_hit"] / v["total"] * 100, 1) if v["total"] else 0.0,
                    "subcategory_acc_pct": round(v["sub_hit"] / v["total"] * 100, 1) if v["total"] else 0.0,
                }
                for cat, v in self.per_cat.items()
            },
        }


# ─────────────────────────────────────────────────────────────
# 8. Batch classification with checkpoint / resume + accuracy
# ─────────────────────────────────────────────────────────────

def classify_batch(
    data_list:       List[Dict],
    api_key:         str,
    base_url:        str,
    model_name:      str,
    temperature:     float,
    checkpoint_file: str,
    max_retries:     int = 3,
    ground_truth:    Optional[List[Tuple[str, str]]] = None,
) -> Tuple[List[Dict], Optional[AccuracyTracker]]:
    done = load_checkpoint(checkpoint_file)
    print(f"  Checkpoint: {len(done):,} items already done, loading from {checkpoint_file}")

    results: List[Dict] = list(done.values())
    todo_indices = [i for i, item in enumerate(data_list)
                    if str(item.get("question_id", "")) not in done]

    if not todo_indices:
        print("  All items already classified. Skipping API calls.")
        # Still build tracker for already-done items if GT provided
        tracker = None
        if ground_truth:
            tracker = AccuracyTracker(ground_truth)
            for global_idx, item in enumerate(data_list):
                qid = str(item.get("question_id", ""))
                if qid in done:
                    clf = done[qid].get("classification", {})
                    tracker.update(global_idx,
                                   clf.get("category", "unknown"),
                                   clf.get("subcategory", "unknown"))
        return results, tracker

    total_todo = len(todo_indices)
    print(f"  Items to classify: {total_todo:,}\n")

    # ── ANSI colors ──
    CAT_COLOR = {
        "Conceptual":   "\033[36m",
        "Debugging":    "\033[33m",
        "Generation":   "\033[32m",
        "Optimization": "\033[35m",
        "parse_error":  "\033[31m",
        "api_error":    "\033[31m",
    }
    RESET = "\033[0m"
    DIM   = "\033[2m"
    GREEN = "\033[32m"
    RED   = "\033[31m"
    BOLD  = "\033[1m"

    # Build tracker over already-checkpointed items first
    tracker: Optional[AccuracyTracker] = None
    if ground_truth:
        tracker = AccuracyTracker(ground_truth)
        for global_idx, item in enumerate(data_list):
            qid = str(item.get("question_id", ""))
            if qid in done:
                clf = done[qid].get("classification", {})
                tracker.update(global_idx,
                               clf.get("category", "unknown"),
                               clf.get("subcategory", "unknown"))

    local_idx_counter = 0  # counts items processed in this session
    for global_idx, item in enumerate(data_list):
        qid = str(item.get("question_id", ""))
        if qid in done:
            continue  # already processed

        local_idx_counter += 1
        classified = classify_one(
            item, api_key, base_url, model_name, temperature, max_retries
        )
        results.append(classified)
        append_checkpoint(checkpoint_file, classified)

        clf    = classified.get("classification", {})
        cat    = clf.get("category",    "unknown")
        sub    = clf.get("subcategory", "unknown")
        reason = clf.get("reason", "")[:100]
        title  = classified.get("title", "N/A")[:70]
        color  = CAT_COLOR.get(cat, "")
        progress = f"[{local_idx_counter:>{len(str(total_todo))}}/{total_todo}]"

        # ── Real-time accuracy update ──
        acc_line = ""
        if tracker is not None:
            result = tracker.update(global_idx, cat, sub)
            if result and not result["excluded"]:
                gt_cat   = result["gt_cat"]
                gt_sub   = result["gt_sub"]
                cat_ok   = result["cat_match"]
                sub_ok   = result["sub_match"]

                cat_mark = f"{GREEN}✓{RESET}" if cat_ok else f"{RED}✗{RESET}"
                sub_mark = f"{GREEN}✓{RESET}" if sub_ok else f"{RED}✗{RESET}"

                cat_acc_str = f"{tracker.cat_accuracy:5.1f}%"
                sub_acc_str = f"{tracker.sub_accuracy:5.1f}%"

                acc_line = (
                    f"           {DIM}GT  category : {gt_cat:<14} {RESET}{cat_mark}  "
                    f"{DIM}GT subcategory: {gt_sub:<22} {RESET}{sub_mark}\n"
                    f"           {BOLD}Running acc → category: {cat_acc_str}  "
                    f"subcategory: {sub_acc_str}  "
                    f"(n={tracker.n_compared}){RESET}"
                )
            elif result and result["excluded"]:
                acc_line = (
                    f"           {RED}[excluded from accuracy: {cat}]{RESET}"
                )

        print(
            f"{DIM}{progress}{RESET} "
            f"{color}{cat:<14}{RESET} "
            f"{DIM}{sub:<22}{RESET} "
            f"{title}\n"
            f"           {DIM}↳ {reason}{RESET}"
        )
        if acc_line:
            print(acc_line)
        print()  # blank line between items

    return results, tracker


# ─────────────────────────────────────────────────────────────
# 8b. Concurrent batch classification
# ─────────────────────────────────────────────────────────────

# Classify
    print(f"\n> Classifying  (model={model_name}  temp={temperature}  workers={max_workers})")
    if max_workers > 1:
        classified, tracker = classify_batch_concurrent(
            data, api_key, base_url, model_name, temperature,
            checkpoint_file, max_retries,
            ground_truth=ground_truth,
            max_workers=max_workers,
            ordered_output=True,   # ← eval 模式：按数据集顺序展示分类结果
        )
    else:
        classified, tracker = classify_batch(
            data, api_key, base_url, model_name, temperature,
            checkpoint_file, max_retries,
            ground_truth=ground_truth,
        )
def classify_batch_concurrent(
    data_list:       List[Dict],
    api_key:         str,
    base_url:        str,
    model_name:      str,
    temperature:     float,
    checkpoint_file: str,
    max_retries:     int = 3,
    ground_truth:    Optional[List[Tuple[str, str]]] = None,
    max_workers:     int = 8,
    ordered_output:  bool = False,
) -> Tuple[List[Dict], Optional[AccuracyTracker]]:
    """
    Thread-pool parallel version of classify_batch().

    Each worker calls classify_one() independently; checkpoint writes,
    AccuracyTracker updates, and console output are all protected by locks
    so no data is lost or corrupted under concurrency.

    Parameters
    ----------
    max_workers    : Number of parallel threads. Set to 1 for serial execution
                     (identical behaviour to classify_batch()).
    ordered_output : If True, classification results are DISPLAYED in dataset
                     order even though API calls still run concurrently. Used
                     by eval mode so console output aligns with the input file.
                     Checkpoint writes still happen as soon as each item
                     completes; only the console output (and the running-
                     accuracy update tied to it) is buffered and ordered.
    """
    done = load_checkpoint(checkpoint_file)
    print(f"  Checkpoint: {len(done):,} items already done, loading from {checkpoint_file}")

    todo: List[Tuple[int, Dict]] = [
        (i, item) for i, item in enumerate(data_list)
        if str(item.get("question_id", "")) not in done
    ]

    # Collect already-done results preserving original order
    results_map: Dict[int, Dict] = {
        i: done[str(item.get("question_id", ""))]
        for i, item in enumerate(data_list)
        if str(item.get("question_id", "")) in done
    }

    if not todo:
        print("  All items already classified. Skipping API calls.")
        tracker = None
        if ground_truth:
            tracker = AccuracyTracker(ground_truth)
            for global_idx, item in enumerate(data_list):
                qid = str(item.get("question_id", ""))
                if qid in done:
                    clf = done[qid].get("classification", {})
                    tracker.update(global_idx,
                                   clf.get("category", "unknown"),
                                   clf.get("subcategory", "unknown"))
        ordered = [results_map[i] for i in range(len(data_list)) if i in results_map]
        return ordered, tracker

    mode_str = "ordered display" if ordered_output else "as-completed display"
    print(f"  Items to classify: {len(todo):,}  |  workers: {max_workers}  |  {mode_str}\n")

    # ── Shared state (all guarded by locks) ──────────────────
    ckpt_lock    = threading.Lock()   # checkpoint file writes
    tracker_lock = threading.Lock()   # AccuracyTracker mutations
    print_lock   = threading.Lock()   # console output
    map_lock     = threading.Lock()   # results_map inserts

    tracker: Optional[AccuracyTracker] = None
    if ground_truth:
        tracker = AccuracyTracker(ground_truth)
        # Seed tracker with already-done items
        for global_idx, item in enumerate(data_list):
            qid = str(item.get("question_id", ""))
            if qid in done:
                clf = done[qid].get("classification", {})
                tracker.update(global_idx,
                               clf.get("category", "unknown"),
                               clf.get("subcategory", "unknown"))

    # ── ANSI colors ──────────────────────────────────────────
    CAT_COLOR = {
        "Conceptual":   "\033[36m",
        "Debugging":    "\033[33m",
        "Generation":   "\033[32m",
        "Optimization": "\033[35m",
        "parse_error":  "\033[31m",
        "api_error":    "\033[31m",
    }
    RESET = "\033[0m"
    DIM   = "\033[2m"
    GREEN = "\033[32m"
    RED   = "\033[31m"
    BOLD  = "\033[1m"

    pbar = tqdm(total=len(todo), desc="Classifying", unit="item", dynamic_ncols=True)

    # Position helpers for [k/N] prefix in ordered mode
    todo_order = [gi for gi, _ in todo]
    pos_of     = {gi: pos for pos, gi in enumerate(todo_order)}
    width      = len(str(len(todo_order)))

    # ── Display + accuracy update for ONE classified item ────
    def _emit(global_idx: int, classified: Dict) -> None:
        clf    = classified.get("classification", {})
        cat    = clf.get("category",    "unknown")
        sub    = clf.get("subcategory", "unknown")
        reason = clf.get("reason", "")[:100]
        title  = classified.get("title", "N/A")[:70]
        color  = CAT_COLOR.get(cat, "")

        acc_line = ""
        if tracker is not None:
            with tracker_lock:
                result = tracker.update(global_idx, cat, sub)
            if result and not result["excluded"]:
                gt_cat = result["gt_cat"]
                gt_sub = result["gt_sub"]
                cat_ok = result["cat_match"]
                sub_ok = result["sub_match"]
                cat_mark = f"{GREEN}✓{RESET}" if cat_ok else f"{RED}✗{RESET}"
                sub_mark = f"{GREEN}✓{RESET}" if sub_ok else f"{RED}✗{RESET}"
                with tracker_lock:
                    cat_acc_str = f"{tracker.cat_accuracy:5.1f}%"
                    sub_acc_str = f"{tracker.sub_accuracy:5.1f}%"
                    n_cmp = tracker.n_compared
                acc_line = (
                    f"           {DIM}GT  category : {gt_cat:<14} {RESET}{cat_mark}  "
                    f"{DIM}GT subcategory: {gt_sub:<22} {RESET}{sub_mark}\n"
                    f"           {BOLD}Running acc → category: {cat_acc_str}  "
                    f"subcategory: {sub_acc_str}  (n={n_cmp}){RESET}"
                )
            elif result and result["excluded"]:
                acc_line = f"           {RED}[excluded from accuracy: {cat}]{RESET}"

        # [k/N] prefix only makes sense when output is ordered
        if ordered_output:
            pos = pos_of.get(global_idx, 0) + 1
            prefix = f"{DIM}[{pos:>{width}}/{len(todo_order)}]{RESET} "
        else:
            prefix = ""

        with print_lock:
            tqdm.write(
                f"{prefix}"
                f"{color}{cat:<14}{RESET} "
                f"{DIM}{sub:<22}{RESET} "
                f"{title}\n"
                f"           {DIM}↳ {reason}{RESET}"
            )
            if acc_line:
                tqdm.write(acc_line)
            tqdm.write("")
            pbar.update(1)

    # ── Worker: only classify + checkpoint + store; NO display ──
    def _worker(global_idx: int, item: Dict) -> Tuple[int, Dict]:
        classified = classify_one(
            item, api_key, base_url, model_name, temperature, max_retries
        )
        with ckpt_lock:
            append_checkpoint(checkpoint_file, classified)
        with map_lock:
            results_map[global_idx] = classified
        return global_idx, classified

    # ── Dispatch work ────────────────────────────────────────
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_worker, global_idx, item): global_idx
            for global_idx, item in todo
        }

        if ordered_output:
            # Buffer completions and emit them in dataset order.
            buffer: Dict[int, Dict] = {}
            next_pos = 0
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc:
                    gi = futures[fut]
                    with print_lock:
                        tqdm.write(f"{RED}[ERROR] item {gi}: {exc}{RESET}")
                    # Insert a placeholder so the ordered drain doesn't stall
                    buffer[gi] = {
                        "title": "N/A",
                        "classification": {
                            "category":    "api_error",
                            "subcategory": "api_error",
                            "reason":      f"Worker raised: {exc}",
                        },
                    }
                else:
                    gi, classified = fut.result()
                    buffer[gi] = classified

                # Drain any contiguous completed items starting at next_pos
                while next_pos < len(todo_order) and todo_order[next_pos] in buffer:
                    gi = todo_order[next_pos]
                    _emit(gi, buffer.pop(gi))
                    next_pos += 1
        else:
            # Original behaviour: emit as soon as each task completes.
            for fut in as_completed(futures):
                exc = fut.exception()
                if exc:
                    gi = futures[fut]
                    with print_lock:
                        tqdm.write(f"{RED}[ERROR] item {gi}: {exc}{RESET}")
                    continue
                gi, classified = fut.result()
                _emit(gi, classified)

    pbar.close()

    # Return results in original dataset order (skip gaps from missing items)
    ordered = [results_map[i] for i in range(len(data_list)) if i in results_map]
    return ordered, tracker

# ─────────────────────────────────────────────────────────────
# 9. Output cleaning
# ─────────────────────────────────────────────────────────────

_ITEM_DROP_FIELDS   = {"questionId", "viewCount", "matchType"}
_ANSWER_DROP_FIELDS = {"answerId", "creationDate", "ownerUserId", "viewCount"}
_CLF_DROP_FIELDS    = {"id"}


def clean_for_output(item: Dict) -> Dict:
    cleaned = {k: v for k, v in item.items() if k not in _ITEM_DROP_FIELDS}
    if "answers" in cleaned and isinstance(cleaned["answers"], list):
        cleaned["answers"] = [
            {k: v for k, v in ans.items() if k not in _ANSWER_DROP_FIELDS}
            for ans in cleaned["answers"]
        ]
    if "classification" in cleaned and isinstance(cleaned["classification"], dict):
        cleaned["classification"] = {
            k: v for k, v in cleaned["classification"].items()
            if k not in _CLF_DROP_FIELDS
        }
    return cleaned


# ─────────────────────────────────────────────────────────────
# 10. Format conversion
# ─────────────────────────────────────────────────────────────

def to_unified_format(
    classified_data:    List[Dict],
    display_model_name: str,
    model_provider:     str,
) -> List[Dict]:
    CATEGORY_MAP = {
        "Conceptual":   "conceptual",
        "Debugging":    "debugging",
        "Generation":   "generation",
        "Optimization": "optimization",
        "parse_error":  "parse_error",
        "api_error":    "api_error",
    }

    unified = []
    for item in tqdm(classified_data, desc="Converting", unit="item"):
        qid = item.get(
            "question_id",
            hashlib.md5(item.get("title", "").encode()).hexdigest(),
        )
        clf     = item.get("classification", {})
        answers = item.get("answers", [])
        answer_content = answers[0].get("content", "") if answers else ""

        unified.append({
            "question": f"{item.get('title', '')}\n{item.get('question_body', '')}",
            "questionMetadata": {
                "type":    CATEGORY_MAP.get(clf.get("category", ""), "other"),
                "subtype": clf.get("subcategory", "general"),
                "tag":     "verilog",
            },
            "answer": answer_content,
            "model":  display_model_name,
            "modelMetadata": {
                "name":     display_model_name,
                "provider": model_provider,
            },
            "classificationReason": clf.get("reason", ""),
        })

    return unified


# ─────────────────────────────────────────────────────────────
# 11. Statistics
# ─────────────────────────────────────────────────────────────

def compute_stats(classified_data: List[Dict]) -> Dict:
    cat_counts: Dict[str, int] = {}
    sub_counts: Dict[str, int] = {}

    for item in classified_data:
        clf = item.get("classification", {})
        cat = clf.get("category", "unknown")
        sub = clf.get("subcategory", "unknown")
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        sub_counts[sub] = sub_counts.get(sub, 0) + 1

    total = len(classified_data)
    error_cats = {"parse_error", "api_error", "unknown"}
    valid = sum(v for k, v in cat_counts.items() if k not in error_cats)

    return {
        "total":                    total,
        "valid_classified":         valid,
        "classification_rate":      valid / total * 100 if total else 0.0,
        "category_distribution":    cat_counts,
        "subcategory_distribution": sub_counts,
    }


# ─────────────────────────────────────────────────────────────
# 12. Visualization
# ─────────────────────────────────────────────────────────────

def plot_stats(stats: Dict, save_path: str = "classification_stats.png") -> None:
    cat_data = stats["category_distribution"]
    sub_data = stats["subcategory_distribution"]
    total    = stats["total"]

    ERROR_CATS  = {"parse_error", "api_error", "unknown"}
    COLOR_OK    = "#1D9E75"
    COLOR_ERR   = "#D85A30"
    COLOR_SPINE = "#cccccc"
    COLOR_TEXT  = "#555555"
    LABEL_SZ    = 9

    def _fmt(x, _):
        return f"{int(x / 1000)}k" if x >= 1000 else str(int(x))

    def _hbar(ax, data: Dict[str, int], title: str):
        pairs = sorted(data.items(), key=lambda kv: kv[1])
        if not pairs:
            ax.set_visible(False)
            return
        labels, vals = zip(*pairs)
        colors = [COLOR_ERR if lb in ERROR_CATS else COLOR_OK for lb in labels]
        bars = ax.barh(list(labels), list(vals), color=colors,
                       height=0.6, edgecolor="none")

        ax.set_title(title, fontsize=11, fontweight="bold", pad=10, loc="left")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt))
        ax.tick_params(axis="y", labelsize=LABEL_SZ)
        ax.tick_params(axis="x", labelsize=LABEL_SZ, colors=COLOR_TEXT)
        ax.yaxis.set_tick_params(labelcolor=COLOR_TEXT)

        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.spines["bottom"].set_visible(True)
        ax.spines["bottom"].set_color(COLOR_SPINE)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.xaxis.grid(True, color=COLOR_SPINE, linewidth=0.5,
                      linestyle="--", alpha=0.7)
        ax.set_axisbelow(True)

        max_val = max(vals) if vals else 1
        for bar, v in zip(bars, vals):
            pct = v / total * 100 if total else 0
            ax.text(
                v + max_val * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{v:,}  ({pct:.1f}%)",
                va="center", ha="left",
                fontsize=LABEL_SZ - 0.5,
                color=COLOR_TEXT,
            )
        ax.set_xlim(0, max_val * 1.4)

    fig_h = max(5, len(sub_data) * 0.5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, fig_h))
    fig.suptitle(
        f"Verilog Classification Statistics  "
        f"(n={total:,}  |  valid rate={stats['classification_rate']:.1f}%)",
        fontsize=13, fontweight="bold", y=1.02,
    )

    _hbar(ax1, cat_data, "Main category")
    _hbar(ax2, sub_data, "Subcategory")

    from matplotlib.patches import Patch
    fig.legend(
        handles=[
            Patch(facecolor=COLOR_OK,  label="Valid classification"),
            Patch(facecolor=COLOR_ERR, label="Error / unparseable"),
        ],
        loc="lower center", ncol=2, fontsize=9,
        frameon=False, bbox_to_anchor=(0.5, -0.05),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Chart saved -> {save_path}")
    plt.show()


# ─────────────────────────────────────────────────────────────
# 13. I/O helpers
# ─────────────────────────────────────────────────────────────

def save_json(data: Any, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(data: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for item in tqdm(data, desc=f"Writing {os.path.basename(path)}", unit="item"):
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def print_examples(classified_data: List[Dict], n: int = 5) -> None:
    print(f"\n-- Classification examples (first {n}) --")
    for i, item in enumerate(classified_data[:n]):
        clf = item.get("classification", {})
        print(f"\n[{i+1}] {item.get('title', 'N/A')}")
        print(f"     category   : {clf.get('category')}")
        print(f"     subcategory: {clf.get('subcategory')}")
        print(f"     reason     : {clf.get('reason', '')[:120]}")


def print_stats(stats: Dict) -> None:
    W = 58
    print("\n" + "=" * W)
    print("  Classification complete")
    print("=" * W)
    print(f"  Total items         : {stats['total']:>8,}")
    print(f"  Valid classified    : {stats['valid_classified']:>8,}")
    print(f"  Classification rate : {stats['classification_rate']:>7.1f}%")

    print("\n  Category distribution:")
    for cat, cnt in sorted(stats["category_distribution"].items(),
                           key=lambda kv: -kv[1]):
        pct = cnt / stats["total"] * 100 if stats["total"] else 0
        bar = "=" * min(int(pct / 2), 30)
        print(f"    {cat:<26s} {cnt:>6,}  {pct:>5.1f}%  {bar}")

    print("\n  Subcategory distribution:")
    for sub, cnt in sorted(stats["subcategory_distribution"].items(),
                           key=lambda kv: -kv[1]):
        pct = cnt / stats["total"] * 100 if stats["total"] else 0
        bar = "=" * min(int(pct / 2), 30)
        print(f"    {sub:<26s} {cnt:>6,}  {pct:>5.1f}%  {bar}")
    print("=" * W)


def print_accuracy_report(tracker: AccuracyTracker) -> None:
    """Print a formatted final accuracy report."""
    W = 58
    s = tracker.summary()
    print("\n" + "=" * W)
    print("  Accuracy Report (vs Ground Truth)")
    print("=" * W)
    print(f"  Items compared      : {s['n_compared']:>8,}")
    print(f"  Errors excluded     : {s['error_excluded']:>8,}")
    print(f"  Category accuracy   : {s['category_accuracy']:>7.2f}%")
    print(f"  Subcategory accuracy: {s['subcategory_accuracy']:>7.2f}%")
    print("\n  Per-category breakdown:")
    for cat, v in sorted(s["per_category"].items()):
        print(
            f"    {cat:<22s}  n={v['total']:<4}  "
            f"cat={v['category_acc_pct']:>5.1f}%  "
            f"sub={v['subcategory_acc_pct']:>5.1f}%"
        )
    print("=" * W)


# ─────────────────────────────────────────────────────────────
# 14. Build pipeline  (eval mode — original, unchanged)
# ─────────────────────────────────────────────────────────────

def build_dataset(
    json_file_path:     str,
    api_key:            str,
    base_url:           str,
    model_name:         str,
    display_model_name: str,
    model_provider:     str,
    temperature:        float,
    checkpoint_file:    str,
    classified_out:     str,
    unified_out:        str,
    summary_out:        str,
    stats_png:          str,
    max_retries:        int = 3,
    ground_truth_file:  Optional[str] = None,
    max_workers:        int = 1,        # ← set > 1 to enable concurrency
) -> None:

    # Load ground truth (optional)
    ground_truth: Optional[List[Tuple[str, str]]] = None
    if ground_truth_file:
        print(f"\n> Loading ground truth from: {ground_truth_file}")
        ground_truth = load_ground_truth(ground_truth_file)

    # Load data
    print(f"\n> Loading data from: {json_file_path}")
    data = load_data(json_file_path)
    print(f"  {len(data):,} items loaded.")

    if ground_truth and len(ground_truth) != len(data):
        print(
            f"  WARNING: ground truth has {len(ground_truth)} rows "
            f"but data has {len(data)} items — accuracy may be partial."
        )

    # Classify
        # Classify
    print(f"\n> Classifying  (model={model_name}  temp={temperature}  workers={max_workers})")
    if max_workers > 1:
            classified, tracker = classify_batch_concurrent(
                data, api_key, base_url, model_name, temperature,
                checkpoint_file, max_retries,
                ground_truth=ground_truth,
                max_workers=max_workers,
                ordered_output=True,  # ← eval 模式：按数据集顺序展示分类结果
            )
    else:
            classified, tracker = classify_batch(
                data, api_key, base_url, model_name, temperature,
                checkpoint_file, max_retries,
                ground_truth=ground_truth,
            )

    # Convert
    print("\n> Converting to unified format ...")
    unified = to_unified_format(classified, display_model_name, model_provider)

    # Stats
    stats = compute_stats(classified)
    print_stats(stats)
    print_examples(classified)

    # Accuracy report
    accuracy_summary: Optional[Dict] = None
    if tracker is not None:
        print_accuracy_report(tracker)
        accuracy_summary = tracker.summary()

    # Save
    print("\n> Saving output files ...")
    cleaned_classified = [clean_for_output(item) for item in classified]
    save_json(cleaned_classified, classified_out)
    print(f"  Classified JSON  -> {classified_out}")

    save_jsonl(unified, unified_out)
    print(f"  Unified JSONL    -> {unified_out}")

    summary = {
        "total_records":            stats["total"],
        "valid_classified":         stats["valid_classified"],
        "classification_rate_pct":  round(stats["classification_rate"], 2),
        "category_distribution":    stats["category_distribution"],
        "subcategory_distribution": stats["subcategory_distribution"],
        "model_used":               display_model_name,
        "model_provider":           model_provider,
        "temperature":              temperature,
    }
    if accuracy_summary:
        summary["accuracy"] = accuracy_summary   # ← accuracy baked into summary JSON

    save_json(summary, summary_out)
    print(f"  Summary JSON     -> {summary_out}")

    # Plot
    print("\n> Generating statistics chart ...")
    plot_stats(stats, save_path=stats_png)


# ─────────────────────────────────────────────────────────────
# 14b. Build pipeline  (full mode — no ground truth, separate outputs)
# ─────────────────────────────────────────────────────────────

def build_dataset_full(
    json_file_path:     str,
    api_key:            str,
    base_url:           str,
    model_name:         str,
    display_model_name: str,
    model_provider:     str,
    temperature:        float,
    checkpoint_file:    str,
    classified_out:     str,
    unified_out:        str,
    summary_out:        str,
    stats_png:          str,
    max_retries:        int = 3,
    max_workers:        int = 8,        # ← higher default for full runs
) -> None:
    """
    Full-dataset classification pipeline.

    Identical to build_dataset() but with ground truth disabled entirely.
    Designed for production runs over the complete, unannotated dataset.
    Separate default output filenames (configured in __main__) prevent
    accidental overwrite of eval-run outputs.

    Parameters
    ----------
    json_file_path     : Path to the full JSONL / JSON dataset.
    api_key            : LLM API key.
    base_url           : LLM API base URL.
    model_name         : Model identifier string sent to the API.
    display_model_name : Human-readable model name written to output files.
    model_provider     : Provider name written to output files.
    temperature        : Sampling temperature (0.0 recommended for determinism).
    checkpoint_file    : JSONL file used for checkpoint / resume.
    classified_out     : Output path for classified JSON.
    unified_out        : Output path for unified JSONL.
    summary_out        : Output path for summary JSON.
    stats_png          : Output path for statistics chart PNG.
    max_retries        : Maximum API retry attempts per item.
    max_workers        : Parallel threads (default 8; set 1 for serial).
    """
    print("\n" + "=" * 60)
    print(f"  MODE: FULL-DATASET CLASSIFICATION  (workers={max_workers})")
    print("=" * 60)

    # Load data
    print(f"\n> Loading data from: {json_file_path}")
    data = load_data(json_file_path)
    print(f"  {len(data):,} items loaded.")

    # Classify (ground_truth=None → no accuracy tracking)
    print(f"\n> Classifying  (model={model_name}  temp={temperature}  workers={max_workers})")
    _batch_fn = classify_batch_concurrent if max_workers > 1 else classify_batch
    classified, _ = _batch_fn(
        data, api_key, base_url, model_name, temperature,
        checkpoint_file, max_retries,
        ground_truth=None,
        **( {"max_workers": max_workers} if max_workers > 1 else {} ),
    )

    # Convert
    print("\n> Converting to unified format ...")
    unified = to_unified_format(classified, display_model_name, model_provider)

    # Stats
    stats = compute_stats(classified)
    print_stats(stats)
    print_examples(classified)

    # Save
    print("\n> Saving output files ...")
    cleaned_classified = [clean_for_output(item) for item in classified]
    save_json(cleaned_classified, classified_out)
    print(f"  Classified JSON  -> {classified_out}")

    save_jsonl(unified, unified_out)
    print(f"  Unified JSONL    -> {unified_out}")

    summary = {
        "total_records":            stats["total"],
        "valid_classified":         stats["valid_classified"],
        "classification_rate_pct":  round(stats["classification_rate"], 2),
        "category_distribution":    stats["category_distribution"],
        "subcategory_distribution": stats["subcategory_distribution"],
        "model_used":               display_model_name,
        "model_provider":           model_provider,
        "temperature":              temperature,
        "mode":                     "full",
    }
    save_json(summary, summary_out)
    print(f"  Summary JSON     -> {summary_out}")

    # Plot
    print("\n> Generating statistics chart ...")
    plot_stats(stats, save_path=stats_png)


# ─────────────────────────────────────────────────────────────
# 15. Entry point  <-- all configuration lives here
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ════════════════════════════════════════════════
    #  RUN MODE
    #  "eval" → small validation dataset + ground truth accuracy
    #  "full" → full production dataset, no ground truth
    #
    #  Override from command line:
    #    python classify_verilog.py --mode full
    #    python classify_verilog.py --mode eval
    # ════════════════════════════════════════════════

    RUN_MODE = "eval"  # default; change to "full" for production run
    if "--mode" in sys.argv:
        idx = sys.argv.index("--mode")
        if idx + 1 < len(sys.argv):
            RUN_MODE = sys.argv[idx + 1].strip().lower()

    if RUN_MODE not in ("eval", "full"):
        raise ValueError(f"Unknown --mode '{RUN_MODE}'. Choose 'eval' or 'full'.")

    print(f"\n  Run mode: {RUN_MODE.upper()}")

    # ════════════════════════════════════════════════
    #  SHARED SETTINGS  (apply to both modes)
    # ════════════════════════════════════════════════

    MODEL_NAME         = ""
    DISPLAY_MODEL_NAME = MODEL_NAME
    MODEL_PROVIDER     = ""
    TEMPERATURE        = 0.0
    MAX_RETRIES        = 5

    # ── Concurrency ──────────────────────────────────────────
    # Number of parallel threads used for API calls.
    #   1  → serial (original behaviour, easiest to debug)
    #   8  → good default for most hosted APIs
    #   16+ → only if the API endpoint supports high concurrency
    # Override from CLI: python classify_verilog.py --mode full --workers 16
    MAX_WORKERS = 8
    if "--workers" in sys.argv:
        idx = sys.argv.index("--workers")
        if idx + 1 < len(sys.argv):
            MAX_WORKERS = int(sys.argv[idx + 1])
    print(f"  Workers   : {MAX_WORKERS}")
    # ─────────────────────────────────────────────────────────

    API_KEY  = os.environ.get("DEEPSEEK_API_KEY", "")

    if not API_KEY:
        raise EnvironmentError(
            "API key not set.\n"
            "  Windows PowerShell : $env:DEEPSEEK_API_KEY = 'sk-xxxxx'\n"
            "  Windows cmd        : set DEEPSEEK_API_KEY=sk-xxxxx\n"
            "  Linux / macOS      : export DEEPSEEK_API_KEY=sk-xxxxx"
        )

    # ════════════════════════════════════════════════
    #  EVAL MODE CONFIG  (small validation dataset)
    # ════════════════════════════════════════════════

    if RUN_MODE == "eval":

        JSON_FILE_PATH    = r""
        GROUND_TRUTH_FILE = r""
        # GROUND_TRUTH_FILE = None   # ← uncomment to disable

        CHECKPOINT_FILE     = "checkpoint-v3.jsonl"
        CLASSIFIED_OUT_FILE = "classified_verilog-v3.json"
        UNIFIED_OUT_FILE    = "unified_verilog-v3.jsonl"
        SUMMARY_FILE        = "classification_summary-v3.json"
        STATS_PNG           = "classification_stats-v3.png"

        build_dataset(
            json_file_path=JSON_FILE_PATH,
            api_key=API_KEY,
            base_url=BASE_URL,
            model_name=MODEL_NAME,
            display_model_name=DISPLAY_MODEL_NAME,
            model_provider=MODEL_PROVIDER,
            temperature=TEMPERATURE,
            checkpoint_file=CHECKPOINT_FILE,
            classified_out=CLASSIFIED_OUT_FILE,
            unified_out=UNIFIED_OUT_FILE,
            summary_out=SUMMARY_FILE,
            stats_png=STATS_PNG,
            max_retries=MAX_RETRIES,
            ground_truth_file=GROUND_TRUTH_FILE,
            max_workers=MAX_WORKERS,
        )

    # ════════════════════════════════════════════════
    #  FULL MODE CONFIG  (complete production dataset)
    # ════════════════════════════════════════════════

    elif RUN_MODE == "full":

        # ── Edit the paths below for your full dataset ──────────
        FULL_JSON_FILE_PATH = r""

        FULL_CHECKPOINT_FILE     = ""
        FULL_CLASSIFIED_OUT_FILE = ""
        FULL_UNIFIED_OUT_FILE    = ""
        FULL_SUMMARY_FILE        = ""
        FULL_STATS_PNG           = ""
        # ────────────────────────────────────────────────────────

        build_dataset_full(
            json_file_path=FULL_JSON_FILE_PATH,
            api_key=API_KEY,
            base_url=BASE_URL,
            model_name=MODEL_NAME,
            display_model_name=DISPLAY_MODEL_NAME,
            model_provider=MODEL_PROVIDER,
            temperature=TEMPERATURE,
            checkpoint_file=FULL_CHECKPOINT_FILE,
            classified_out=FULL_CLASSIFIED_OUT_FILE,
            unified_out=FULL_UNIFIED_OUT_FILE,
            summary_out=FULL_SUMMARY_FILE,
            stats_png=FULL_STATS_PNG,
            max_retries=MAX_RETRIES,
            max_workers=MAX_WORKERS,
        )