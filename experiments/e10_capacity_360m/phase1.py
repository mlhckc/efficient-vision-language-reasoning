"""E10 Phase-1 execution guardrails.

Phase 0 measured the 360M readout family, recommended a per-cell wall and a
per-identity ceiling, and deliberately left both unset. Phase 1 sets them and
makes them enforceable. Nothing here authorises a scientific cell: the B4/B4r
core matrix stays refused while no valid external scientific core authorization
grant exists, and every entry point in this module refuses with it. Phase 3
moved that decision out of the source constant and into the external grant; the
constant is a legacy refusal sentinel that must remain None.

The independent Phase-0 review found one execution-readiness gap: the E10
recipe drops the inherited E8B ``wall_clock_halt_hours`` field and no runner
re-imposed an E10-specific wall. The repair is here rather than in the frozen
recipe, so the twelve recipe digests in the Phase-0 recipe contract are
unchanged. The wall is carried by a signed, process-bound cell permit and is
re-checked on every optimizer step.

Examples, from the project root with the project environment sourced:

    python -B -m experiments.e10_capacity_360m.run phase1-policy
    python -B -m experiments.e10_capacity_360m.run phase1-verify
"""

from __future__ import annotations

import ast
import inspect
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import config  # noqa: E402
from experiments.e10_capacity_360m import e10_common as e10  # noqa: E402


TASK_ID = e10.PHASE1_TASK_ID

# The whole cell lifecycle, not only training. Every one of these stages must
# run inside ScientificCellGuard.stage(), and a cell that leaves any of them
# unguarded can never be recorded as completed.
CELL_STAGES = (
    "setup",
    "training",
    "development_evaluation",
    "g14_diagnostics",
    "final_r1_evaluation",
    "result_publication",
)

# The user's 2026-08-10 Phase-1 resource decisions, in the words that bind.
USER_DECISIONS = {
    "per_cell_wall": (
        "E10_PER_CELL_WALL_CLOCK_HOURS = 12.0 as a hard executable operational "
        "halt, not a projection field; a cell that reaches it fails closed, is "
        "recorded as halted with correct accounting, and never continues"
    ),
    "per_identity_ceiling": (
        "E10_PER_IDENTITY_CEILING_HOURS = 40.0, never more permissive than the "
        "programme-wide frozen-model identity ceiling; every gate uses "
        "min(E10_PER_IDENTITY_CEILING_HOURS, PER_MODEL_IDENTITY_CEILING_HOURS)"
    ),
    "retry_policy": (
        "no automatic retry; the previous nominal one-forced-retry behaviour is "
        "withdrawn because the Gate-1 stress projection leaves operationally "
        "meaningless remaining margins, so a failed scientific cell requires a "
        "fresh explicit user or reviewer retry authorization record"
    ),
    "a5_a8c": (
        "OPEN_UNQUANTIFIED_RESERVE retained; A5/A8c are outside E10 and remain "
        "re-deferred, but if they use the same pinned pretrained SmolLM2-360M "
        "frozen identity their future compute is accounted against the same "
        "authoritative frozen-model identity ceiling, not as free independent "
        "budget"
    ),
    "storage": (
        "physical fit is established; a formal project storage allocation "
        "remains unestablished and free space is not an allocation claim"
    ),
}

# Sources that must never contain an automatic-retry path. Tests are excluded
# by design: a test may fabricate an authorization record inside a temporary
# directory, which is exactly what the production sources may not do.
_SCANNED_ROOTS = ("experiments", "src")
# Third-party code vendored for scoring only. It is not a project execution
# path and it is not Python 3, so it is excluded from the source scan.
_EXCLUDED_PARTS = frozenset({"vendor", "__pycache__"})
_WRITE_CALLS = frozenset({
    "atomic_write_json", "atomic_replace_json", "write_text", "write_bytes",
    "open", "touch", "mkdir", "rename", "replace", "dump", "symlink_to",
})
_RETRY_AUTHORIZATION_TOKENS = ("retry_authorization", "RETRY_AUTHORIZATION")


class GuardrailError(AssertionError):
    """A Phase-1 guardrail refused."""


CellWallExceeded = e10.CellWallExceeded


# --- resource policy ---------------------------------------------------------

def per_cell_wall_hours() -> float:
    return e10.per_cell_wall_hours()


def effective_identity_ceiling_hours() -> float:
    return e10.effective_identity_ceiling_hours()


def resource_policy() -> dict:
    return e10.resource_policy()


# --- frozen execution plan ---------------------------------------------------

def execution_plan() -> list[tuple[str, str, int]]:
    """The frozen twelve-cell order.

    Deliberately parameterless and deliberately free of any reference to a
    result, a metric, a ledger or a checkpoint, so no scheduling decision can
    ever be conditioned on how an earlier cell turned out.
    """
    return e10.pair_preserving_order()


def assert_frozen_execution_plan() -> dict:
    """Prove the plan is fixed, unconditional and pair-preserving."""
    signature = inspect.signature(execution_plan)
    if signature.parameters:
        raise GuardrailError("the E10 execution plan takes an argument")
    tree = ast.parse(Path(__file__).read_text(), filename=__file__)
    definitions = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "execution_plan"
    ]
    if len(definitions) != 1:
        raise GuardrailError("execution_plan is not defined exactly once")
    for call in (node for node in ast.walk(definitions[0])
                 if isinstance(node, ast.Call)):
        if _dotted_name(call.func) != "e10.pair_preserving_order":
            raise GuardrailError(
                f"execution_plan calls {_dotted_name(call.func)!r}; only the "
                f"frozen pair-preserving order may decide the order"
            )
    order = execution_plan()
    if len(order) != 12 or len(set(order)) != 12:
        raise GuardrailError("the E10 execution plan is not the twelve cells")
    if order != list(e10.CORE_CELLS):
        raise GuardrailError("the E10 execution plan drifted from CORE_CELLS")
    for scale in e10.CORE_SCALES:
        for seed in e10.CORE_SEEDS:
            present = [arm for arm, s, d in order if s == scale and d == seed]
            if present != list(e10.CORE_ARMS):
                raise GuardrailError(
                    f"B4/B4r pair preservation broken at {scale}/seed{seed}"
                )
    return {"cells": len(order), "pair_preserving": True, "adaptive": False}


# --- no automatic retry ------------------------------------------------------

def _dotted_name(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _scanned_sources() -> list[Path]:
    paths = [PROJECT_ROOT / "config.py"]
    for root in _SCANNED_ROOTS:
        paths.extend(sorted((PROJECT_ROOT / root).rglob("*.py")))
    return [
        path for path in paths
        if path.is_file()
        and not _EXCLUDED_PARTS.intersection(path.relative_to(PROJECT_ROOT).parts)
    ]


def _mentions_retry_authorization(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if isinstance(child, (ast.Name, ast.Attribute)):
            text = _dotted_name(child)
        elif isinstance(child, ast.Constant) and isinstance(child.value, str):
            text = child.value
        else:
            continue
        if any(token in text for token in _RETRY_AUTHORIZATION_TOKENS):
            return True
    return False


def assert_no_automatic_retry() -> dict:
    """No project source may create a retry authorization or consume a retry.

    Together with e10_common.record_forced_retry, which refuses unless a fresh
    single-use authorization record for exactly that cell already exists, this
    makes an automatic retry impossible rather than merely discouraged.
    """
    if config.E10_AUTOMATIC_RETRIES_PER_IDENTITY != 0:
        raise GuardrailError(
            "E10_AUTOMATIC_RETRIES_PER_IDENTITY must be 0; it is "
            f"{config.E10_AUTOMATIC_RETRIES_PER_IDENTITY!r}"
        )
    if e10.AUTOMATIC_RETRIES_PER_IDENTITY != 0:
        raise GuardrailError("the E10 automatic-retry allowance is not zero")
    writers = []
    callers = []
    scanned = _scanned_sources()
    for path in scanned:
        tree = ast.parse(path.read_text(), filename=str(path))
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            name = _dotted_name(call.func)
            if name.split(".")[-1] == "record_forced_retry":
                callers.append(str(path.relative_to(PROJECT_ROOT)))
            if name.split(".")[-1] not in _WRITE_CALLS:
                continue
            arguments = list(call.args) + [keyword.value for keyword in call.keywords]
            if any(_mentions_retry_authorization(argument) for argument in arguments):
                writers.append(str(path.relative_to(PROJECT_ROOT)))
    if writers:
        raise GuardrailError(
            f"project source writes a retry authorization record: {writers}"
        )
    if callers:
        raise GuardrailError(
            f"project source consumes a retry automatically: {callers}"
        )
    return {
        "files_scanned": len(scanned),
        "automatic_retry_writers": 0,
        "automatic_retry_call_sites": 0,
        "automatic_retries_per_identity": 0,
    }


# --- scientific refusal ------------------------------------------------------

def scientific_refusal() -> str | None:
    """The refusal message for the first frozen cell, or None if it would run."""
    arm, scale, seed = execution_plan()[0]
    try:
        e10.assert_core_entry_authorized(arm, scale, seed)
    except SystemExit as error:
        return str(error)
    except AssertionError as error:
        return f"E10 CORE REFUSED: {error}"
    return None


def assert_scientific_core_refused() -> dict:
    refusal = scientific_refusal()
    if refusal is None:
        raise GuardrailError(
            "the E10 scientific core did not refuse; Phase 1 authorises no cell"
        )
    return {
        "refused": True,
        "refusal": refusal,
        "e10_training_authorized": e10.E10_TRAINING_AUTHORIZED,
        "core_authorization_grant_present": (
            e10.core_authorization_grant_path().is_file()
        ),
    }


def assert_cell_entry_guarded(arm: str, scale: str, seed: int) -> dict:
    """The mandatory Phase-1 entry check for one scientific cell."""
    if (arm, scale, seed) not in e10.CORE_CELLS:
        raise GuardrailError(f"{arm}/{scale}/seed{seed} is not an E10 core cell")
    assert_frozen_execution_plan()
    assert_no_automatic_retry()
    policy = resource_policy()
    e10.assert_core_entry_authorized(arm, scale, seed)
    return policy


# --- the per-cell wall -------------------------------------------------------

@dataclass
class _CellState:
    started_monotonic_ns: int
    started_utc: str
    permit: e10.CellExecutionPermit


class ScientificCellGuard:
    """Hold one scientific cell inside its mandatory whole-cell wall.

    The wall governs the entire cell lifecycle, not only training. Two
    mechanisms make that unavoidable rather than optional:

    1. Every scientifically relevant stage in CELL_STAGES must run inside
       ``stage()``, which checks the deadline on entry and on exit. A cell
       that leaves any stage unguarded can never be recorded as completed.
    2. Finalisation re-derives the current monotonic time against the permit
       deadline on every exit path, including a normal exit in which nothing
       checked the wall. A crossing forces the wall-clock halt outcome,
       latches the halt and raises.

    The guard does not interrupt an in-flight operation. A CUDA kernel, an
    evaluation pass or a generation tail that is already running continues
    until it returns. What the guard guarantees is that such a cell cannot
    publish a successful scientific result, cannot be charged as completed,
    and cannot continue authorised scientific work under that permit.
    """

    def __init__(self, arm: str, scale: str, seed: int, *,
                 started_monotonic_ns: int | None = None,
                 clock: Callable[[], int] = time.monotonic_ns,
                 utc: Callable[[], str] = e10.utc_now) -> None:
        if (arm, scale, seed) not in e10.CORE_CELLS:
            raise GuardrailError(
                f"{arm}/{scale}/seed{seed} is not an E10 core cell"
            )
        self.arm = arm
        self.scale = scale
        self.seed = seed
        self.identity = e10.model_identity(arm)
        self._clock = clock
        self._utc = utc
        self._requested_start_ns = started_monotonic_ns
        self._state: _CellState | None = None
        self._halted_reason: str | None = None
        self._finalized: dict | None = None
        self._finalize_attempted = False
        self._stages_completed: set[str] = set()

    # -- lifecycle ------------------------------------------------------------

    def __enter__(self) -> "ScientificCellGuard":
        policy = assert_cell_entry_guarded(self.arm, self.scale, self.seed)
        now_ns = self._clock()
        start_ns = now_ns if self._requested_start_ns is None \
            else int(self._requested_start_ns)
        if start_ns > now_ns:
            raise GuardrailError(
                "a cell start in the future would extend the per-cell wall"
            )
        permit = e10.authorize_cell_execution(
            self.arm, self.scale, self.seed, started_monotonic_ns=start_ns
        )
        self.policy = policy
        self._state = _CellState(
            started_monotonic_ns=start_ns,
            started_utc=self._utc(),
            permit=permit,
        )
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self._state is not None and not self._finalize_attempted:
            outcome = "completed" if exc_type is None else "terminated"
            self._finalize(outcome)
        if exc_type is None and self._halted_reason is not None:
            # A crossing that nothing observed during the cell still has to
            # surface: a silent return would read as a successful cell.
            raise CellWallExceeded(self._halted_reason)
        return False

    # -- enforcement ----------------------------------------------------------

    @property
    def halted(self) -> bool:
        return self._halted_reason is not None

    @property
    def permit(self) -> e10.CellExecutionPermit:
        return self._require_open().permit

    @property
    def stages_completed(self) -> tuple[str, ...]:
        return tuple(stage for stage in CELL_STAGES
                     if stage in self._stages_completed)

    def _require_open(self) -> _CellState:
        if self._state is None:
            raise GuardrailError("the E10 cell guard was not entered")
        if self._halted_reason is not None:
            raise CellWallExceeded(self._halted_reason)
        if self._finalize_attempted:
            raise GuardrailError(
                "the E10 cell was already finalised; it cannot continue"
            )
        return self._state

    def elapsed_seconds(self) -> float:
        state = self._state
        if state is None:
            return 0.0
        return (self._clock() - state.started_monotonic_ns) / 1_000_000_000.0

    def _deadline_reached(self, now_ns: int | None = None) -> bool:
        state = self._state
        if state is None:
            return False
        now_ns = self._clock() if now_ns is None else now_ns
        return now_ns >= state.permit.deadline_monotonic_ns

    def check(self, context: str = "step", *,
              revalidate_grant: bool = False) -> dict:
        """Re-check the wall. Raises once reached, and stays raised.

        ``revalidate_grant`` additionally re-reads the external scientific
        authorization grant, so a grant removed or altered under a running cell
        stops it at the next stage boundary.
        """
        state = self._require_open()
        try:
            return e10.validate_cell_permit(
                state.permit, revalidate_grant=revalidate_grant
            )
        except CellWallExceeded as error:
            self._halt(f"{error} (at {context})")
            raise

    @contextmanager
    def stage(self, name: str):
        """Run one scientifically relevant stage inside the whole-cell wall.

        The deadline is checked when the stage opens and again when it closes,
        and only a stage that closed inside the wall counts towards the
        completeness requirement that a successful cell must satisfy. The
        external authorization grant is revalidated on entry, so a stage can
        never open under an authorization that has since been withdrawn.
        """
        if name not in CELL_STAGES:
            raise GuardrailError(
                f"{name!r} is not an E10 cell stage; expected one of "
                f"{list(CELL_STAGES)}"
            )
        self.check(f"stage_enter:{name}", revalidate_grant=True)
        yield self
        self.check(f"stage_exit:{name}")
        self._stages_completed.add(name)

    def guarded_optimizer_step(self, optimizer) -> dict:
        """Step only through the single authorised optimizer gate."""
        state = self._require_open()
        try:
            return e10.guarded_optimizer_step(optimizer, state.permit)
        except CellWallExceeded as error:
            self._halt(f"{error} (at optimizer_step)")
            raise

    def _halt(self, reason: str) -> None:
        if self._halted_reason is None:
            self._halted_reason = reason
            if not self._finalize_attempted:
                self._finalize("wall_clock_halted", detail=reason)

    # -- accounting -----------------------------------------------------------

    def _finalize(self, outcome: str, detail: str | None = None) -> dict:
        state = self._state
        if state is None:  # pragma: no cover - guarded by _require_open
            raise GuardrailError("cannot finalize a cell that never opened")
        # One attempt only. A failed attempt has already latched shared state,
        # and retrying it here would hide that latch behind a second error.
        self._finalize_attempted = True
        now_ns = self._clock()
        occupancy_ns = max(0, now_ns - state.started_monotonic_ns)
        # Fail closed on EVERY exit path. The deadline is re-derived here even
        # when no optimizer step and no voluntary check ran during the overrun,
        # so an unguarded evaluation, G14 or publication tail that crossed the
        # wall can never be finalised as a completed cell.
        if outcome != "wall_clock_halted" and self._deadline_reached(now_ns):
            crossing = (
                f"E10 per-cell wall of {state.permit.per_cell_wall_hours} hours "
                f"was reached before finalisation of "
                f"{self.arm}/{self.scale}/seed{self.seed}"
            )
            detail = f"{detail}; {crossing}" if detail else crossing
            outcome = "wall_clock_halted"
        if outcome == "wall_clock_halted" and self._halted_reason is None:
            self._halted_reason = detail
        # A successful cell must prove every scientifically relevant stage ran
        # inside the wall. An unguarded stage is an unmeasured stage.
        if outcome == "completed":
            missing = [stage for stage in CELL_STAGES
                       if stage not in self._stages_completed]
            if missing:
                outcome = "terminated"
                unguarded = f"unguarded scientific stages: {missing}"
                detail = f"{detail}; {unguarded}" if detail else unguarded
        try:
            charge = e10.charge_core_cell_hours(
                state.permit,
                occupancy_ns=occupancy_ns,
                outcome=outcome,
                started_utc=state.started_utc,
                ended_utc=self._utc(),
            )
            record = self._write_cell_record(outcome, detail, charge)
        except BaseException as error:
            # Accounting is never skipped silently: an unreconciled charge
            # latches shared state and blocks every later E10 operation.
            e10.latch_shared_failure(
                f"e10_core_cell_{outcome}",
                error,
                claim_sha256=None,
                occupancy_ns=occupancy_ns,
            )
            raise
        self._finalized = {"outcome": outcome, "charge": charge, "record": record}
        return self._finalized

    def _write_cell_record(self, outcome: str, detail: str | None,
                           charge: dict) -> dict:
        stamp = self._utc().replace("-", "").replace(":", "")
        path = e10.OUT_DIR / (
            f"cell_{outcome}_{self.arm}_{self.scale}_seed{self.seed}_{stamp}.json"
        )
        payload = {
            "schema_version": 1,
            "record_type": "e10_scientific_cell_outcome",
            "task_id": TASK_ID,
            "status": outcome.upper(),
            "utc": self._utc(),
            "binding": e10.binding_record(),
            "cell": [self.arm, self.scale, self.seed],
            "identity": self.identity,
            "recipe_sha256": charge["recipe_digests"][
                f"{self.arm}_{self.scale}_seed{self.seed}"
            ],
            "per_cell_wall_clock_hours": self.policy["per_cell_wall_clock_hours"],
            "effective_identity_ceiling_hours": self.policy[
                "effective_identity_ceiling_hours"
            ],
            "gpu_occupancy_ns": charge["gpu_occupancy_ns"],
            "gpu_hours": charge["gpu_hours"],
            "spend_entry_id": charge["entry_id"],
            "detail": detail,
            "scientific_success": outcome == "completed",
            "guarded_stages_completed": list(self.stages_completed),
            "required_stages": list(CELL_STAGES),
            "wall_reached_before_finalisation": self._deadline_reached(),
            "automatic_retry_available": False,
            "retry_requires_fresh_authorization": True,
        }
        digest = e10.atomic_write_json(path, payload)
        return {"path": _relative(path), "sha256": digest}


# --- reporting ---------------------------------------------------------------

def policy_record() -> dict:
    return {
        "task_id": TASK_ID,
        "policy": resource_policy(),
        "user_decisions": USER_DECISIONS,
        "execution_plan": [list(cell) for cell in execution_plan()],
        "frozen_plan": assert_frozen_execution_plan(),
        "no_automatic_retry": assert_no_automatic_retry(),
        "scientific_core": assert_scientific_core_refused(),
        "wall_reimposition": wall_reimposition_record(),
    }


def wall_reimposition_record() -> dict:
    """Show the recipe stays frozen and the wall is imposed outside it."""
    recipes = {}
    for arm, scale, seed in execution_plan():
        recipe = e10.build_recipe(arm, scale, seed)
        if "wall_clock_halt_hours" in recipe:
            raise GuardrailError(
                "the frozen E10 recipe must not carry a resource field"
            )
        recipes[f"{arm}_{scale}_seed{seed}"] = e10.recipe_sha256(recipe)
    return {
        "recipe_carries_wall_field": False,
        "wall_imposed_by": (
            "experiments.e10_capacity_360m.e10_common.CellExecutionPermit via "
            "ScientificCellGuard"
        ),
        "wall_source": "config.E10_PER_CELL_WALL_CLOCK_HOURS",
        "per_cell_wall_clock_hours": per_cell_wall_hours(),
        "scope": "whole cell lifecycle, not only training",
        "required_guarded_stages": list(CELL_STAGES),
        "enforcement_points": [
            "stage entry and stage exit for every stage in CELL_STAGES",
            "every guarded optimizer step",
            "finalisation on every exit path, including a normal exit in "
            "which nothing checked the wall",
            "charge_core_cell_hours refuses a completed outcome above the wall",
            "the spend-ledger validator refuses a completed core entry above "
            "the wall on read",
        ],
        "interrupts_in_flight_operations": False,
        "guarantee": (
            "a cell that crosses the wall cannot publish a successful "
            "scientific result, cannot be charged as completed, and cannot "
            "continue authorised scientific work under that permit"
        ),
        "recipe_digests": recipes,
    }


def _relative(path: Path) -> str:
    if path.is_relative_to(PROJECT_ROOT):
        return str(path.relative_to(PROJECT_ROOT))
    return str(path)


def _reference(path: Path) -> dict:
    return {"path": _relative(path), "sha256": e10.sha256_file(path)}


def _phase0_preserved_records() -> dict:
    preserved = {}
    for filename in e10.PHASE0_RECORD_SHA256:
        repository = e10.OUT_DIR / filename
        shared = e10.SHARED_STATE_DIR / filename
        if repository.is_file():
            path, location = repository, "repository"
        elif shared.is_file():
            path, location = shared, "shared_state"
        else:
            raise GuardrailError(f"Phase-0 record is missing: {filename}")
        digest = e10.sha256_file(path)
        if digest != e10.PHASE0_RECORD_SHA256[filename]:
            raise GuardrailError(f"Phase-0 record changed: {filename}")
        preserved[filename] = {"location": location, "sha256": digest}
    return preserved


def write_phase1_records() -> dict:
    """Write the three immutable Phase-1 governance records, once."""
    existing = [str(path) for path in e10.PHASE1_PATHS if path.exists()]
    if existing:
        raise FileExistsError(f"immutable E10 Phase-1 records exist: {existing}")
    e10.assert_no_embargo_reference()
    assert_scientific_core_refused()

    amendment = {
        "schema_version": 1,
        "record_type": "e10_phase1_binding_amendment",
        "task_id": TASK_ID,
        "status": "PHASE0_EVIDENCE_CARRIED_FORWARD",
        "NON_SCIENTIFIC": True,
        "utc": e10.utc_now(),
        "binding": e10.binding_record(),
        "reason": (
            "Phase 1 sets the two mandatory E10 resource constants and adds the "
            "Phase-1 guard sources, so the live source and config digests move. "
            "The Phase-0 records are preserved byte for byte and are carried "
            "forward by this record instead of being rewritten."
        ),
        "change_scope": {
            "resource_constants_set": True,
            "scientific_recipe_changed": False,
            "core_matrix_changed": False,
            "model_pin_changed": False,
            "phase0_record_rewritten": False,
        },
        "phase0_source_digest": e10.PHASE0_SOURCE_DIGEST,
        "phase0_config_digest": e10.PHASE0_CONFIG_DIGEST,
        "precorrection_source_digest": e10.PRECORRECTION_SOURCE_DIGEST,
        "precorrection_config_digest": e10.PRECORRECTION_CONFIG_DIGEST,
        "preserved_records": _phase0_preserved_records(),
        "scientific_execution": {
            "optimizer_steps": 0,
            "scientific_cells_executed": 0,
            "gpu_hours_charged": 0.0,
            "e10_training_authorized": None,
        },
    }
    amendment_sha = e10.atomic_write_json(e10.PHASE1_AMENDMENT_PATH, amendment)

    policy = {
        "schema_version": 1,
        "record_type": "e10_phase1_resource_policy",
        "task_id": TASK_ID,
        "status": "GUARDRAILS_FROZEN_SCIENTIFIC_EXECUTION_STILL_REFUSED",
        "NON_SCIENTIFIC": True,
        "utc": e10.utc_now(),
        "git_commit": e10.current_git_commit(),
        "binding": e10.binding_record(),
        "decided_by": "user",
        "decided_utc_date": "2026-08-10",
        "user_decisions": USER_DECISIONS,
        "policy": resource_policy(),
        "gate1_basis": {
            "path": str(e10.GATE1_PATH.relative_to(PROJECT_ROOT)),
            "sha256": e10.sha256_file(e10.GATE1_PATH),
            "recommended_per_cell_wall_hours": 8.194757662528623,
            "largest_stress_cell_hours": 7.879574675508292,
            "chosen_wall_hours": per_cell_wall_hours(),
            "post_retry_margin_hours": {
                "pretrained_smollm2_360m": 0.04064131283994055,
                "random_smollm2_360m": 0.007927129566605906,
            },
            "margin_interpretation": (
                "operationally meaningless, which is why no automatic retry "
                "exists and a retry needs a fresh explicit authorization"
            ),
        },
        "frozen_matrix": {
            "arms": list(e10.CORE_ARMS),
            "scales": list(e10.CORE_SCALES),
            "seeds": list(e10.CORE_SEEDS),
            "cells": len(e10.CORE_CELLS),
            "order": [list(cell) for cell in execution_plan()],
            "pair_preservation": "B4 with B4r, inseparable, at every scale and seed",
            "changed_by_this_task": False,
        },
        "wall_reimposition": wall_reimposition_record(),
        "no_automatic_retry": assert_no_automatic_retry(),
        "retry_authorization_contract": {
            "record_type": "e10_retry_authorization",
            "written_by": sorted(e10.RETRY_AUTHORIZED_BY),
            "written_by_code": False,
            "single_use": True,
            "bound_to": "one arm, scale, seed, reason and failure record",
            "max_authorized_per_identity": e10.MAX_FORCED_RETRIES_PER_IDENTITY,
        },
        "storage": {
            "physical_fit": True,
            "formal_project_allocation": "no established project allocation",
            "qualification": (
                "free filesystem space is a physical-fit observation, not "
                "evidence of an approved project storage allocation"
            ),
        },
        "scientific_execution": assert_scientific_core_refused(),
        "clean_test_accessed": False,
    }
    policy_sha = e10.atomic_write_json(e10.PHASE1_POLICY_PATH, policy)

    a5_a8c = {
        "schema_version": 1,
        "record_type": "e10_phase1_a5_a8c_identity_governance",
        "task_id": TASK_ID,
        "status": "OPEN_UNQUANTIFIED_RESERVE",
        "NON_SCIENTIFIC": True,
        "utc": e10.utc_now(),
        "binding": e10.binding_record(),
        "decided_by": "user",
        "decision": USER_DECISIONS["a5_a8c"],
        "outside_e10": True,
        "re_deferred": True,
        "cancelled": False,
        "implemented_by_this_task": False,
        "run_by_this_task": False,
        "numeric_reserve_invented": False,
        "numeric_reserve_hours": None,
        "shared_identity_governance": {
            "condition": (
                "if A5 or A8c uses the same pinned pretrained SmolLM2-360M "
                "frozen identity as B4"
            ),
            "consequence": (
                "their future compute is charged against the same authoritative "
                "frozen-model identity aggregate and the same effective ceiling; "
                "it is never treated as free independent budget"
            ),
            "authoritative_ceiling_constant": (
                "config.PER_MODEL_IDENTITY_CEILING_HOURS"
            ),
            "effective_ceiling_rule": resource_policy()["ceiling_rule"],
            "shared_identity": "pretrained_smollm2_360m",
        },
        "phase0_disposition": _reference(e10.A5_A8C_DISPOSITION_PATH),
    }
    a5_a8c_sha = e10.atomic_write_json(e10.PHASE1_A5_A8C_PATH, a5_a8c)
    return {
        "phase1_binding_amendment": amendment_sha,
        "phase1_resource_policy": policy_sha,
        "phase1_a5_a8c_identity_governance": a5_a8c_sha,
    }


def write_phase1_repair_record() -> dict:
    """Write the immutable whole-cell wall repair record, once."""
    if e10.PHASE1_REPAIR_PATH.exists():
        raise FileExistsError(
            f"immutable E10 Phase-1 repair record exists: {e10.PHASE1_REPAIR_PATH}"
        )
    e10.assert_no_embargo_reference()
    assert_scientific_core_refused()
    preserved = {}
    for filename, expected in e10.PHASE1_R1_RECORD_SHA256.items():
        path = e10.OUT_DIR / filename
        digest = e10.sha256_file(path)
        if digest != expected:
            raise GuardrailError(f"Phase-1 revision-1 record changed: {filename}")
        preserved[filename] = {"location": "repository", "sha256": digest}
    repair = {
        "schema_version": 1,
        "record_type": "e10_phase1_whole_cell_wall_repair",
        "task_id": TASK_ID,
        "status": "WHOLE_CELL_WALL_ENFORCED",
        "NON_SCIENTIFIC": True,
        "utc": e10.utc_now(),
        "binding": e10.binding_record(),
        "reason": (
            "The independent Phase-1 review returned CHANGES_REQUIRED on one "
            "scientific-execution blocker: the 12-hour limit was not a true "
            "whole-cell hard wall. A cell could finish training inside the "
            "wall, run an unguarded evaluation or G14 tail, cross the wall "
            "with no optimizer step and no voluntary check, and still exit "
            "with outcome completed and an over-wall charge."
        ),
        "review": {
            "verdict": "CHANGES_REQUIRED",
            "reviewed_head": "175d856bcb754e666bf448fd25b77313d5676870",
            "blocking_defect": (
                "the per-cell wall governed only guarded interactions, not the "
                "whole cell lifecycle"
            ),
        },
        "change_scope": {
            "resource_constants_changed": False,
            "scientific_recipe_changed": False,
            "core_matrix_changed": False,
            "model_pin_changed": False,
            "phase0_record_rewritten": False,
            "phase1_record_rewritten": False,
        },
        "phase1_r1_source_digest": e10.PHASE1_R1_SOURCE_DIGEST,
        "phase1_r1_config_digest": e10.PHASE1_R1_CONFIG_DIGEST,
        "preserved_records": preserved,
        "regenerated_within_open_revision": {
            "regenerated": True,
            "superseded_sha256": e10.PHASE1_REPAIR_SUPERSEDED_SHA256,
            "reason": (
                "A follow-up review of the same open repair revision required "
                "the accounting backstops to halt at equality with the wall, "
                "not only above it. That edit moved the live source digest, so "
                "this record was regenerated inside the same unapproved "
                "revision instead of chaining a second amendment onto an "
                "amendment. Its content is otherwise unchanged, the superseded "
                "version remains in git history at commit a4bb773, and no "
                "Phase-0 record and no Phase-1 revision-1 record was touched."
            ),
        },
        "whole_cell_wall": wall_reimposition_record(),
        "scientific_execution": {
            "optimizer_steps": 0,
            "scientific_cells_executed": 0,
            "gpu_hours_charged": 0.0,
            "e10_training_authorized": None,
        },
    }
    digest = e10.atomic_write_json(e10.PHASE1_REPAIR_PATH, repair)
    return {"phase1_whole_cell_wall_repair": digest}


def _validated_policy_record() -> dict:
    record = e10.read_json_mapping(e10.PHASE1_POLICY_PATH)
    e10.assert_recorded_binding(
        record, "E10 Phase-1 resource policy", e10.PHASE1_POLICY_PATH
    )
    if record.get("record_type") != "e10_phase1_resource_policy" \
            or record.get("task_id") != TASK_ID \
            or record.get("NON_SCIENTIFIC") is not True \
            or record.get("status") != (
                "GUARDRAILS_FROZEN_SCIENTIFIC_EXECUTION_STILL_REFUSED"
            ):
        raise GuardrailError("Phase-1 resource policy semantics mismatch")
    if record.get("policy") != resource_policy():
        raise GuardrailError("Phase-1 recorded policy differs from the live one")
    if record.get("frozen_matrix", {}).get("order") != [
        list(cell) for cell in execution_plan()
    ]:
        raise GuardrailError("Phase-1 recorded core order differs from the live one")
    if record.get("scientific_execution", {}).get("refused") is not True:
        raise GuardrailError("Phase-1 policy does not record the core refusal")
    return record


def _validated_a5_a8c_record() -> dict:
    record = e10.read_json_mapping(e10.PHASE1_A5_A8C_PATH)
    e10.assert_recorded_binding(
        record, "E10 Phase-1 A5/A8c governance", e10.PHASE1_A5_A8C_PATH
    )
    if record.get("record_type") != "e10_phase1_a5_a8c_identity_governance" \
            or record.get("task_id") != TASK_ID \
            or record.get("status") != "OPEN_UNQUANTIFIED_RESERVE" \
            or record.get("numeric_reserve_hours") is not None \
            or record.get("numeric_reserve_invented") is not False \
            or record.get("cancelled") is not False \
            or record.get("re_deferred") is not True:
        raise GuardrailError("Phase-1 A5/A8c governance semantics mismatch")
    shared = record.get("shared_identity_governance", {})
    if shared.get("authoritative_ceiling_constant") != (
        "config.PER_MODEL_IDENTITY_CEILING_HOURS"
    ) or shared.get("shared_identity") != "pretrained_smollm2_360m":
        raise GuardrailError("Phase-1 A5/A8c shared-identity clause mismatch")
    return record


def verify() -> dict:
    """Validate every Phase-1 guardrail and its recorded evidence."""
    repair = e10.validate_phase1_repair()
    amendment = e10.validate_phase1_amendment()
    policy = _validated_policy_record()
    a5_a8c = _validated_a5_a8c_record()
    frozen = {}
    for name, expected in e10.FROZEN_RESULT_BASELINES.items():
        root = config.RESULTS_DIR / "experiments" / name
        rows = sorted(
            (item.relative_to(PROJECT_ROOT).as_posix(), e10.sha256_file(item))
            for item in root.rglob("*") if item.is_file()
        )
        stream = "".join(f"{digest}  {relative}\n" for relative, digest in rows)
        observed = {
            "files": len(rows),
            "tree_sha256": e10.sha256_bytes(stream.encode("utf-8")),
        }
        if observed != expected:
            raise GuardrailError(f"frozen {name} result tree changed: {observed}")
        frozen[name] = observed
    return {
        "phase1_whole_cell_wall_repair": {
            "sha256": e10.sha256_file(e10.PHASE1_REPAIR_PATH),
            "status": repair["status"],
            "preserved_records": len(repair["preserved_records"]),
        },
        "whole_cell_wall": wall_reimposition_record(),
        "phase1_binding_amendment": {
            "sha256": e10.sha256_file(e10.PHASE1_AMENDMENT_PATH),
            "preserved_records": len(amendment["preserved_records"]),
        },
        "phase1_resource_policy": {
            "sha256": e10.sha256_file(e10.PHASE1_POLICY_PATH),
            "policy": policy["policy"],
        },
        "phase1_a5_a8c_identity_governance": {
            "sha256": e10.sha256_file(e10.PHASE1_A5_A8C_PATH),
            "status": a5_a8c["status"],
        },
        "phase0_records_revalidated": _revalidate_phase0_records(),
        "frozen_result_trees": frozen,
        "no_automatic_retry": assert_no_automatic_retry(),
        "frozen_plan": assert_frozen_execution_plan(),
        "scientific_core": assert_scientific_core_refused(),
        "no_scientific_outputs": e10.assert_no_scientific_outputs(),
    }


def _revalidate_phase0_records() -> dict:
    """Every Phase-0 consumer path still validates across the amendment."""
    from experiments.e10_capacity_360m import projection

    return {
        "governance": sorted(projection.validate_governance_records()),
        "model_verification": e10.validate_model_verification()["status"],
        "source_correction": e10.validate_source_correction()["status"],
        "calibration_revoked": e10.calibration_is_revoked(),
        "shared_state": e10.assert_shared_state_healthy(
            require_calibration_complete=True
        )["calibration_complete"],
    }


def main(argv: Any = None) -> int:  # pragma: no cover - thin CLI wrapper
    import json

    print(json.dumps(verify(), indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
