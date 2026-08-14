"""E7b serial-benchmark tests: gates, registry, timing math, immutability.

CPU-only: no GPU work, no encoder forward pass, no store opened. Every
load-bearing gate is exercised as a known-positive and a known-negative,
following the project convention that a check that cannot fail is not
evidence. The embargoed clean-test target is never read or named.
"""

from __future__ import annotations

import ast
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.e7b_serial_efficiency import run as e7b  # noqa: E402

VERBOSE = False

_CHECKS: list[tuple[str, bool]] = []
_INSIDE_KNOWN_NEGATIVE = False


def check(name: str, condition: bool, detail: str = "") -> None:
    if not _INSIDE_KNOWN_NEGATIVE:
        _CHECKS.append((name, bool(condition)))
        if VERBOSE:
            print(f"  [{'PASS' if condition else 'FAIL'}] {name}"
                  + (f" — {detail}" if detail else ""))
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def must_fail(name: str, thunk) -> None:
    global _INSIDE_KNOWN_NEGATIVE
    _INSIDE_KNOWN_NEGATIVE = True
    try:
        thunk()
        raised = False
    except (AssertionError, KeyError, RuntimeError, ValueError, IndexError,
            TypeError, SystemExit, FileNotFoundError):
        raised = True
    finally:
        _INSIDE_KNOWN_NEGATIVE = False
    check(f"{name} (known-negative raises)", raised,
          "" if raised else "the unsafe path was NOT refused")


# --- Registry integrity -------------------------------------------------------

def test_registry() -> None:
    check("eight systems registered", len(e7b.SYSTEMS) == 8)
    check("question_only is the only context-only system",
          [n for n, s in e7b.SYSTEMS.items() if s["context_only"]]
          == ["question_only"])
    check("the fp16-cached class is exactly the three token systems",
          sorted(n for n, s in e7b.SYSTEMS.items() if s["fp16_cached"])
          == ["e8a_a0p", "e8a_a1", "reasoner"])
    for name, spec in e7b.SYSTEMS.items():
        checkpoint = PROJECT_ROOT / spec["checkpoint"]
        check(f"checkpoint exists for {name}", checkpoint.exists(),
              spec["checkpoint"])
        check(f"seed-0 policy in {name}'s checkpoint name",
              "seed0" in checkpoint.name)
        value = e7b.stored_accuracy(spec["seed0_accuracy"])
        check(f"stored seed-0 accuracy resolves for {name}",
              0.30 < value < 0.70, str(value))
        context = e7b.multiseed_context(spec["multiseed"])
        check(f"multi-seed context resolves for {name}",
              0.30 < context["mean"] < 0.70 and context["std"] >= 0)
    must_fail("an unknown accuracy route fails closed",
              lambda: e7b.stored_accuracy(("results/experiments/"
                                           "v2_07_scaling/results.json",
                                           "no_such_key")))


# --- Stage-name contracts -----------------------------------------------------

def test_stage_names() -> None:
    multimodal = e7b.build_pipeline("fusion")
    check("global multimodal pipeline has nine stages (no projection)",
          len(multimodal.stage_names) == 9
          and "S8_projection" not in multimodal.stage_names)
    floor = e7b.build_pipeline("question_only")
    check("question_only pipeline has five stages (no image branch)",
          len(floor.stage_names) == 5
          and floor.stage_names[0] == "S5_tokenise")
    reasoner = e7b.build_pipeline("reasoner")
    check("reasoner pipeline has nine stages (no projection)",
          len(reasoner.stage_names) == 9
          and "S8_projection" not in reasoner.stage_names)
    a1 = e7b.build_pipeline("e8a_a1")
    check("A1 pipeline has ten stages including the projection",
          len(a1.stage_names) == 10 and "S8_projection" in a1.stage_names)
    check("every pipeline ends at answer decode",
          all(p.stage_names[-1] == "S10_decode"
              for p in (multimodal, floor, reasoner, a1)))


# --- Reproduction gate --------------------------------------------------------

def _rows(cached_index, serial_index, cached_logits, serial_logits):
    serial = {"q1": {"index": serial_index, "answer": "yes",
                     "imageId": "img1", "logits": serial_logits}}
    cached = {"q1": {"index": cached_index, "logits": cached_logits}}
    return serial, cached


def test_reproduction_gate() -> None:
    agree_serial, agree_cached = _rows(2, 2, [0.0, 0.1, 0.9],
                                       [0.0, 0.1, 0.9])
    record = e7b.reproduction_gate("fusion", agree_serial, agree_cached)
    check("exact agreement passes the fp32 gate",
          record["agreements"] == 1 and not record["disagreements"])

    flip_serial, flip_cached = _rows(2, 1, [0.0, 0.1, 0.9],
                                     [0.0, 0.9, 0.1])
    must_fail("any fp32 disagreement halts",
              lambda: e7b.reproduction_gate("fusion", flip_serial,
                                            flip_cached))

    # fp16 class: a near tie within twice the logit perturbation is
    # explained; a large-margin flip is not.
    near_serial, near_cached = _rows(2, 1, [0.0, 0.5000, 0.5002],
                                     [0.0, 0.5002, 0.5000])
    original_floor = e7b.FP16_AGREEMENT_FLOOR
    try:
        e7b.FP16_AGREEMENT_FLOOR = 0
        record = e7b.reproduction_gate("reasoner", near_serial, near_cached)
        check("a documented near tie is explained for the fp16 class",
              record["disagreements"][0]["explained"]
              and record["disagreements"][0]["max_abs_logit_delta"] > 0)
        far_serial, far_cached = _rows(2, 1, [0.0, 0.1, 5.0],
                                       [0.0, 5.0, 0.1])
        must_fail("an unexplained large-margin flip halts even for fp16",
                  lambda: e7b.reproduction_gate("reasoner", far_serial,
                                                far_cached))
    finally:
        e7b.FP16_AGREEMENT_FLOOR = original_floor

    many_serial = {f"q{i}": {"index": 1, "answer": "a", "imageId": "x",
                             "logits": [0.0, 1.0]} for i in range(64)}
    many_cached = {f"q{i}": {"index": 0, "logits": [1.0, 0.0]}
                   for i in range(64)}
    must_fail("falling below the agreement floor halts",
              lambda: e7b.reproduction_gate("reasoner", many_serial,
                                            many_cached))


# --- Timing and maths ---------------------------------------------------------

def test_timing_helpers() -> None:
    floor = e7b.timed_calls(lambda: None, 5, 20)
    check("timed_calls returns the full statistic set",
          all(k in floor for k in ("median_ms", "mean_ms", "std_ms",
                                   "p5_ms", "p95_ms", "n")))
    check("a no-op times at the floor", floor["median_ms"] < 0.05,
          str(floor["median_ms"]))
    check("common-denominator arithmetic",
          round(4462 / e7b.RAW_DEV_QUESTIONS, 5) == 0.44602)

    rng_a = np.random.default_rng(e7b.ORDER_SEED)
    rng_b = np.random.default_rng(e7b.ORDER_SEED)
    orders_a = [list(rng_a.permutation(8)) for _ in range(3)]
    orders_b = [list(rng_b.permutation(8)) for _ in range(3)]
    check("pass-order randomisation is pinned and reproducible",
          orders_a == orders_b and len({tuple(o) for o in orders_a}) > 1)


# --- Rows and immutability ----------------------------------------------------

def test_rows_and_outputs() -> None:
    if e7b.ROWS_FILE.exists():
        record = json.loads(e7b.ROWS_FILE.read_text())
        check("pinned row file has the pinned count and seed",
              record["n_rows"] == e7b.N_ROWS
              and record["row_seed"] == e7b.ROW_SEED)
        check("row digest is stable",
              e7b.rows_digest(record) == e7b.rows_digest(
                  json.loads(e7b.ROWS_FILE.read_text())))
    else:
        check("row file not yet built (preflight will build it)", True)

    with tempfile.TemporaryDirectory() as scratch:
        target = Path(scratch) / "record.json"
        e7b.atomic_write_json(target, {"a": 1})
        check("atomic write produces the record",
              json.loads(target.read_text()) == {"a": 1})
        must_fail("overwriting an existing record is refused",
                  lambda: e7b.atomic_write_json(target, {"a": 2}))
        check("the refused overwrite left the original intact",
              json.loads(target.read_text()) == {"a": 1})


# --- N-1 guard over the accepted serial peak-memory boundary ------------------
#
# The accepted E7b source repair (verdict E7B_SOURCE_REPAIR_PASS) moved the
# two CUDA peak reads above the segmented, answer-sweep, cached and batch-64
# regimes, and hoisted one no_grad over the cold query and the determinism
# repeat. Nothing in the suite pinned that structure, so a later edit could
# reintroduce the contaminated boundary while all 55 existing checks still
# passed. The checks below read the current run.py and pin the invariant.
#
# The guard is built from source TEXT, not from the imported module, so the
# same invariants run unchanged against deliberately mutated in-memory
# fixtures. That is what makes it evidence: every invariant is shown to fail
# under a representative violation and to survive benign unrelated edits. The
# live run.py is never modified, and no benchmark entry point is called.

E7B_RUN_SOURCE = (PROJECT_ROOT / "experiments" / "e7b_serial_efficiency"
                  / "run.py")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _dotted(node: ast.AST) -> str:
    """Dotted source name of an attribute chain, or "" when not one."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return ""


def _bound_name(value) -> str | None:
    """The variable a record slot reads from: x, or round(x, n)."""
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Call):
        for argument in value.args:
            if isinstance(argument, ast.Name):
                return argument.id
    return None


class _ChildRun:
    """Structural view of child_run, derived by AST from source text.

    Ordering uses the index of the enclosing top-level statement of
    child_run, then line and column. That is a relative, semantic order:
    it survives edits elsewhere in the file, unlike a fixed line number.
    """

    def __init__(self, source: str) -> None:
        found = [node for node in ast.walk(ast.parse(source))
                 if isinstance(node, ast.FunctionDef)
                 and node.name == "child_run"]
        _require(len(found) == 1,
                 f"expected exactly one child_run, found {len(found)}")
        self.fn = found[0]
        self.pos = {}
        for index, statement in enumerate(self.fn.body):
            for node in ast.walk(statement):
                self.pos[id(node)] = (index, getattr(node, "lineno", 0),
                                      getattr(node, "col_offset", 0))
        self.no_grad = self._no_grad_ids()
        self.record_keys = self._dict_keys(self._record_dict())
        memory = self.record_keys.get("memory")
        _require(isinstance(memory, ast.Dict),
                 "the record's memory slot is not a dict literal")
        self.memory_keys = self._dict_keys(memory)

        self.serial_calls = self._sorted(self._method("serial_query"))
        self.alloc_reads = self._sorted(
            self._api("torch.cuda.max_memory_allocated"))
        self.reserved_reads = self._sorted(
            self._api("torch.cuda.max_memory_reserved"))
        self.resets = self._sorted(
            self._api("torch.cuda.reset_peak_memory_stats"))
        self.empty_cache = self._api("torch.cuda.empty_cache")

        self.warm_name = _bound_name(
            self.record_keys.get("warm_single_unsegmented"))
        self.primary_candidates = [
            node for node in self._assigns_to(self.warm_name)
            if isinstance(node.value, ast.Call)
            and _dotted(node.value.func) == "timed_calls"]
        self.primary = (self.primary_candidates[0]
                        if len(self.primary_candidates) == 1 else None)

        inner = self._primary_callable_ids()
        self.primary_query_calls = [call for call in self.serial_calls
                                    if id(call) in inner]
        outer = [call for call in self.serial_calls if id(call) not in inner]
        if self.primary is None:
            self.pre_calls, self.post_calls = [], []
        else:
            mark = self.pos[id(self.primary)]
            self.pre_calls = [c for c in outer if self.pos[id(c)] < mark]
            self.post_calls = [c for c in outer if self.pos[id(c)] > mark]

        self.peak_alloc_name = self._assigned_name(self.alloc_reads)
        self.peak_reserved_name = self._assigned_name(self.reserved_reads)

    # -- extraction helpers --

    def _sorted(self, nodes: list) -> list:
        return sorted(nodes, key=lambda node: self.pos[id(node)])

    @staticmethod
    def _dict_keys(node: ast.Dict) -> dict:
        return {key.value: value
                for key, value in zip(node.keys, node.values)
                if isinstance(key, ast.Constant) and isinstance(key.value, str)}

    def _record_dict(self) -> ast.Dict:
        for node in ast.walk(self.fn):
            if isinstance(node, ast.Dict) and "memory" in self._dict_keys(node):
                return node
        raise AssertionError("no child_run record dict carrying a memory slot")

    def _no_grad_ids(self) -> set:
        """Every node lexically inside a `with torch.no_grad():` body."""
        guarded = set()
        for node in ast.walk(self.fn):
            if not isinstance(node, ast.With):
                continue
            if not any(isinstance(item.context_expr, ast.Call)
                       and _dotted(item.context_expr.func) == "torch.no_grad"
                       for item in node.items):
                continue
            for statement in node.body:
                guarded.update(id(inner) for inner in ast.walk(statement))
        return guarded

    def _api(self, dotted: str) -> list:
        return [node for node in ast.walk(self.fn)
                if isinstance(node, ast.Call) and _dotted(node.func) == dotted]

    def _method(self, attribute: str) -> list:
        return [node for node in ast.walk(self.fn)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == attribute]

    def _assigns_to(self, name: str | None) -> list:
        if name is None:
            return []
        return self._sorted([
            node for node in ast.walk(self.fn) if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name
                    for target in node.targets)])

    def _assign_of(self, call: ast.Call):
        for node in ast.walk(self.fn):
            if isinstance(node, ast.Assign) and any(
                    inner is call for inner in ast.walk(node.value)):
                return node
        return None

    def _assigned_name(self, reads: list) -> str | None:
        if not reads:
            return None
        assign = self._assign_of(reads[0])
        if assign is None:
            return None
        for target in assign.targets:
            if isinstance(target, ast.Name):
                return target.id
        return None

    def _primary_callable_ids(self) -> set:
        """Nodes inside the callable the primary timing actually times."""
        if self.primary is None or not self.primary.value.args:
            return set()
        first = self.primary.value.args[0]
        target = first
        if isinstance(first, ast.Name):
            defs = [node for node in ast.walk(self.fn)
                    if isinstance(node, ast.FunctionDef)
                    and node.name == first.id]
            if not defs:
                return set()
            target = defs[0]
        return {id(node) for node in ast.walk(target)}

    @staticmethod
    def _segmented(call: ast.Call) -> bool:
        return any(keyword.arg == "segmented" for keyword in call.keywords)

    def later_regimes(self) -> list:
        """Every regime that must not contaminate the serial peak."""
        marks = []
        segmented = [c for c in self.serial_calls if self._segmented(c)]
        if segmented:
            marks.append(("first segmented serial_query",
                          min(self.pos[id(c)] for c in segmented)))
        for key, label in (
                ("serial_rows", "auxiliary answer-sweep regime"),
                ("cached_image_question_side", "cached-image regime"),
                ("cached_feature_head_only", "cached-feature regime"),
                ("batched_image_branch", "batch-64 benchmark")):
            assigns = self._assigns_to(_bound_name(self.record_keys.get(key)))
            if assigns:
                marks.append((label, self.pos[id(assigns[0])]))
        for call in self._method("encode_image_path"):
            marks.append(("cached-image encode_image_path",
                          self.pos[id(call)]))
        for call in self._method("cached_feature_logits"):
            marks.append(("cached_feature_logits call", self.pos[id(call)]))
        for node in ast.walk(self.fn):
            if isinstance(node, ast.Name) and node.id in (
                    "BENCH_BATCH", "WARMUP_BATCH", "ITERS_BATCH"):
                marks.append((f"batch-64 constant {node.id}",
                              self.pos[id(node)]))
        return marks


# -- the nine pinned invariants --

def _n1a(guard: _ChildRun) -> None:
    _require(guard.primary is not None,
             "no primary serial timing marker, so the cold call cannot be "
             "distinguished from later serial_query calls")
    _require(len(guard.pre_calls) == 2,
             "expected exactly two pre-warm-up serial_query calls (the cold "
             f"query and the determinism repeat), found {len(guard.pre_calls)}")
    cold = guard.pre_calls[0]
    _require(not guard._segmented(cold),
             "the cold batch-1 query is segmented")
    _require(id(cold) in guard.no_grad,
             "the cold batch-1 pipeline.serial_query(rows[0]) does NOT "
             "execute inside torch.no_grad()")


def _n1b(guard: _ChildRun) -> None:
    _require(len(guard.pre_calls) == 2,
             "the determinism repeat is not identifiable: expected two "
             f"pre-warm-up serial_query calls, found {len(guard.pre_calls)}")
    cold, repeat = guard.pre_calls
    _require(bool(cold.args) and bool(repeat.args)
             and ast.dump(cold.args[0]) == ast.dump(repeat.args[0]),
             "the second pre-warm-up query is not an immediate duplicate of "
             "the cold query's row")
    _require(id(repeat) in guard.no_grad,
             "the determinism-gate repeat query does NOT execute inside "
             "torch.no_grad()")


def _n1c(guard: _ChildRun) -> None:
    _require(guard.warm_name is not None,
             "the record does not bind warm_single_unsegmented to a variable")
    _require(len(guard.primary_candidates) == 1,
             "expected exactly one timed_calls assignment to "
             f"{guard.warm_name}, found {len(guard.primary_candidates)}")
    _require(bool(guard.primary_query_calls),
             "the primary timed callable never calls serial_query")
    _require(not any(guard._segmented(c) for c in guard.primary_query_calls),
             "the primary serial timing runs segmented queries")


def _n1d(guard: _ChildRun) -> None:
    _require(guard.primary is not None, "no primary serial timing marker")
    _require(bool(guard.alloc_reads) and bool(guard.reserved_reads),
             "child_run does not read both CUDA peak counters")
    mark = guard.pos[id(guard.primary)]
    for label, reads in (("max_memory_allocated", guard.alloc_reads),
                         ("max_memory_reserved", guard.reserved_reads)):
        for read in reads:
            _require(guard.pos[id(read)] > mark,
                     f"torch.cuda.{label}() is read at or before the primary "
                     "serial timed_calls assignment")


def _n1e(guard: _ChildRun) -> None:
    reads = guard.alloc_reads + guard.reserved_reads
    _require(bool(reads), "child_run reads no CUDA peak counter")
    last = max(guard.pos[id(read)] for read in reads)
    for label, at in guard.later_regimes():
        _require(at > last,
                 f"{label} occurs at or before the reported peak reads and "
                 "can contaminate the serial peak")
    for call in guard.post_calls:
        _require(guard.pos[id(call)] > last,
                 "a non-primary serial_query runs before the peak reads")


def _n1f(guard: _ChildRun) -> None:
    _require(len(guard.resets) == 1,
             "torch.cuda.reset_peak_memory_stats() must occur exactly once "
             f"in child_run, found {len(guard.resets)}")
    at = guard.pos[id(guard.resets[0])]
    _require(bool(guard.pre_calls), "no cold serial inference to order against")
    _require(at < guard.pos[id(guard.pre_calls[0])],
             "the peak reset does not precede the cold serial inference")
    for read in guard.alloc_reads + guard.reserved_reads:
        _require(at < guard.pos[id(read)],
                 "the peak reset does not precede the peak reads")


def _n1g(guard: _ChildRun) -> None:
    _require(not guard.empty_cache,
             "torch.cuda.empty_cache() appears inside child_run; cache "
             "manipulation in the serial measurement path changes the "
             "reported reserved figure")


def _n1h(guard: _ChildRun) -> None:
    _require(len(guard.alloc_reads) == 1,
             "torch.cuda.max_memory_allocated() must be read exactly once in "
             f"child_run, found {len(guard.alloc_reads)}: a later read can "
             "overwrite the reported serial peak")
    _require(len(guard.reserved_reads) == 1,
             "torch.cuda.max_memory_reserved() must be read exactly once in "
             f"child_run, found {len(guard.reserved_reads)}")
    for name in (guard.peak_alloc_name, guard.peak_reserved_name):
        _require(name is not None,
                 "a peak read is not bound to a plain variable")
        _require(len(guard._assigns_to(name)) == 1,
                 f"{name} is assigned more than once in child_run")


def _n1i(guard: _ChildRun) -> None:
    for key, expected in (("peak_allocated_mib", guard.peak_alloc_name),
                          ("peak_reserved_mib", guard.peak_reserved_name)):
        _require(key in guard.memory_keys,
                 f"the record's memory slot lost {key}")
        actual = _bound_name(guard.memory_keys[key])
        _require(actual is not None and actual == expected,
                 f"memory.{key} reads from {actual}, not from {expected}, "
                 "the variable assigned at the accepted serial boundary")


_INVARIANTS = (
    ("N1-A", "the cold batch-1 query runs inside torch.no_grad()", _n1a),
    ("N1-B", "the determinism repeat runs inside torch.no_grad()", _n1b),
    ("N1-C", "the primary serial timed_calls marker is identifiable", _n1c),
    ("N1-D", "both peak reads follow the primary serial timing", _n1d),
    ("N1-E", "both peak reads precede every later regime", _n1e),
    ("N1-F", "exactly one peak reset, before the cold query and the reads",
     _n1f),
    ("N1-G", "no empty_cache in child_run's serial measurement path", _n1g),
    ("N1-H", "no later peak read can overwrite the reported values", _n1h),
    ("N1-I", "the record binds the boundary-assigned peak variables", _n1i),
)


def _run_invariant(guard: _ChildRun, invariant) -> tuple[bool, str]:
    try:
        invariant(guard)
    except AssertionError as error:
        return False, str(error)
    return True, ""


def test_source_invariants() -> None:
    guard = _ChildRun(E7B_RUN_SOURCE.read_text())
    for label, description, invariant in _INVARIANTS:
        passed, detail = _run_invariant(guard, invariant)
        check(f"{label}: {description}", passed, detail)
    check("the reported memory schema is unchanged",
          set(guard.memory_keys) == {"baseline_after_load_mib",
                                     "peak_allocated_mib",
                                     "peak_reserved_mib", "cpu"},
          str(sorted(guard.memory_keys)))
    # N1-G is scoped to child_run, so the legitimate preflight empty_cache
    # outside it is not forbidden. Recorded, not asserted, because a benign
    # edit may remove that unrelated call.
    module_calls = len([node for node in ast.walk(
        ast.parse(E7B_RUN_SOURCE.read_text()))
        if isinstance(node, ast.Call)
        and _dotted(node.func) == "torch.cuda.empty_cache"])
    check("the empty_cache guard is scoped to child_run",
          len(guard.empty_cache) == 0,
          f"module-wide empty_cache calls: {module_calls}")


# --- Non-vacuity: the guard must fail under real violations -------------------
#
# A synthetic child_run that satisfies every invariant, mutated one defect at
# a time. run.py itself is never modified.

_FIXTURE = '''
import torch


def child_run(name, pass_index, out_path):
    pipeline = build_pipeline(name)
    rows = load_rows()
    torch.cuda.reset_peak_memory_stats()
    baseline_mib = torch.cuda.memory_allocated() / 2 ** 20
    with torch.no_grad():
        cold_start = time.perf_counter()
        index0, answer0, logits0, _, _ = pipeline.serial_query(rows[0])
        cold_ms = (time.perf_counter() - cold_start) * 1000
        index_repeat, _, logits_repeat, _, _ = pipeline.serial_query(rows[0])
    if index_repeat != index0:
        sys.exit("DETERMINISM GATE FAILED")
    cycle = [rows[i % len(rows)] for i in range(ITERS_SINGLE)]

    def one_query():
        pipeline.serial_query(cycle[0])

    NAME_WARM = timed_calls(one_query, WARMUP_SINGLE, ITERS_SINGLE)
    NAME_ALLOC = torch.cuda.max_memory_allocated() / 2 ** 20
    NAME_RESERVED = torch.cuda.max_memory_reserved() / 2 ** 20
    with torch.no_grad():
        for i in range(SEGMENTED_ITERS):
            _, _, _, _, stages = pipeline.serial_query(rows[i], segmented=True)
    serial_rows = {}
    with torch.no_grad():
        for row in rows:
            index, answer, logits, _, _ = pipeline.serial_query(row)
            serial_rows[row["questionId"]] = {"index": index}
    cached_image = pipeline.encode_image_path(rows[0]["image_path"])
    cached_image_timing = timed_calls(question_side_query, WARMUP_SINGLE,
                                      ITERS_SINGLE)
    cached_feature_timing = timed_calls(
        lambda: pipeline.cached_feature_logits(cached_image, cached_question),
        WARMUP_SINGLE, ITERS_SINGLE)
    batch_rows = rows[:BENCH_BATCH]
    batched_timing = timed_calls(batched_query, WARMUP_BATCH, ITERS_BATCH)
    record = {
        "cold_first_query_ms": round(cold_ms, 2),
        "warm_single_unsegmented": NAME_WARM,
        "cached_image_question_side": cached_image_timing,
        "cached_feature_head_only": cached_feature_timing,
        "batched_image_branch": batched_timing,
        "memory": {"baseline_after_load_mib": round(baseline_mib, 1),
                   "peak_allocated_mib": round(NAME_ALLOC, 1),
                   "peak_reserved_mib": round(NAME_RESERVED, 1),
                   "cpu": cpu_memory_mib()},
        "serial_rows": serial_rows,
    }
    atomic_write_json(out_path, record)
    return 0
'''


def _render(warm: str = "warm", alloc: str = "peak_alloc",
            reserved: str = "peak_reserved") -> str:
    return (_FIXTURE.replace("NAME_WARM", warm)
            .replace("NAME_ALLOC", alloc).replace("NAME_RESERVED", reserved))


_COLD_BLOCK = """    with torch.no_grad():
        cold_start = time.perf_counter()
        index0, answer0, logits0, _, _ = pipeline.serial_query(rows[0])
        cold_ms = (time.perf_counter() - cold_start) * 1000
        index_repeat, _, logits_repeat, _, _ = pipeline.serial_query(rows[0])
"""

_COLD_OUTSIDE = """    cold_start = time.perf_counter()
    index0, answer0, logits0, _, _ = pipeline.serial_query(rows[0])
    cold_ms = (time.perf_counter() - cold_start) * 1000
    with torch.no_grad():
        index_repeat, _, logits_repeat, _, _ = pipeline.serial_query(rows[0])
"""

_REPEAT_OUTSIDE = """    with torch.no_grad():
        cold_start = time.perf_counter()
        index0, answer0, logits0, _, _ = pipeline.serial_query(rows[0])
        cold_ms = (time.perf_counter() - cold_start) * 1000
    index_repeat, _, logits_repeat, _, _ = pipeline.serial_query(rows[0])
"""

_PEAKS = ("    peak_alloc = torch.cuda.max_memory_allocated() / 2 ** 20\n"
          "    peak_reserved = torch.cuda.max_memory_reserved() / 2 ** 20\n")
_WARM_LINE = ("    warm = timed_calls(one_query, WARMUP_SINGLE, "
              "ITERS_SINGLE)\n")
_SWEEP_LINE = "    serial_rows = {}\n"
_RECORD_LINE = "    record = {\n"


def _moved(anchor: str) -> str:
    """The peak reads removed from the boundary and re-inserted at anchor."""
    return _render().replace(_PEAKS, "").replace(anchor, _PEAKS + anchor)


def _mutations() -> tuple:
    return (
        ("N1-A", "the cold query moved outside no_grad",
         _render().replace(_COLD_BLOCK, _COLD_OUTSIDE)),
        ("N1-B", "the determinism repeat moved outside no_grad",
         _render().replace(_COLD_BLOCK, _REPEAT_OUTSIDE)),
        ("N1-C", "the primary timed_calls marker removed",
         _render().replace(_WARM_LINE, '    warm = {"median_ms": 0.0}\n')),
        ("N1-D", "a peak read hoisted above the primary serial timing",
         _moved(_WARM_LINE)),
        ("N1-E", "the peak reads moved below the segmented regime",
         _moved(_SWEEP_LINE)),
        ("N1-E", "the peak reads moved below the batch-64 benchmark",
         _moved(_RECORD_LINE)),
        ("N1-F", "a second reset_peak_memory_stats introduced",
         _render().replace(
             _PEAKS, "    torch.cuda.reset_peak_memory_stats()\n" + _PEAKS)),
        ("N1-G", "empty_cache introduced in the serial path",
         _render().replace(_PEAKS, "    torch.cuda.empty_cache()\n" + _PEAKS)),
        ("N1-H", "a later peak read overwriting the reported value",
         _render().replace(
             _RECORD_LINE,
             "    peak_alloc = torch.cuda.max_memory_allocated() / 2 ** 20\n"
             + _RECORD_LINE)),
        ("N1-I", "the record rebound away from the boundary variable",
         _render().replace("round(peak_alloc, 1)", "round(baseline_mib, 1)")),
    )


def _benign() -> tuple:
    return (
        ("renaming the peak variables",
         _render(alloc="serial_peak_allocated", reserved="serial_peak_res")),
        ("renaming the primary timing variable",
         _render(warm="primary_serial_timing")),
        ("an unrelated statement after the boundary",
         _render().replace(_PEAKS, _PEAKS + '    print("progress")\n')),
        ("an unrelated new record key",
         _render().replace('        "serial_rows": serial_rows,\n',
                           '        "serial_rows": serial_rows,\n'
                           '        "note": "unrelated",\n')),
    )


def test_guard_is_not_vacuous() -> None:
    by_label = {label: invariant for label, _, invariant in _INVARIANTS}
    baseline = _ChildRun(_render())
    for label, description, invariant in _INVARIANTS:
        passed, detail = _run_invariant(baseline, invariant)
        check(f"{label} holds in the unmutated fixture ({description})",
              passed, detail)
    # Each fixture must be a real, parseable, applied mutation. Without
    # this, a replacement that silently matched nothing, or one that
    # produced unparseable source, would satisfy must_fail for the wrong
    # reason and the evidence would be worthless.
    unmutated = _render()
    for label, name, source in _mutations():
        ast.parse(source)
        check(f"the fixture for {name} is a real applied mutation",
              source != unmutated,
              "" if source != unmutated else "the replacement matched nothing")
        must_fail(f"{label} rejects {name}",
                  lambda s=source, i=by_label[label]: i(_ChildRun(s)))
    for name, source in _benign():
        ast.parse(source)
        check(f"the fixture for {name} is a real applied edit",
              source != unmutated,
              "" if source != unmutated else "the replacement matched nothing")
        guard = _ChildRun(source)
        broken = [label for label, _, invariant in _INVARIANTS
                  if not _run_invariant(guard, invariant)[0]]
        check(f"the guard tolerates {name}", not broken, str(broken))


def run() -> None:
    test_registry()
    test_stage_names()
    test_reproduction_gate()
    test_timing_helpers()
    test_rows_and_outputs()
    test_source_invariants()
    test_guard_is_not_vacuous()
    failed = [name for name, passed in _CHECKS if not passed]
    if failed:
        raise AssertionError(f"{len(failed)} check(s) failed: {failed}")
    print(f"  test_e7b: {len(_CHECKS)} checks passed")


if __name__ == "__main__":
    VERBOSE = True
    run()
