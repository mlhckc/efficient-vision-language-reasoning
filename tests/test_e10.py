"""CPU-only contract tests for E10 Phase-0 calibration and refusal gates."""

from __future__ import annotations

import ast
import copy
import hashlib
import hmac
import json
import multiprocessing
import os
import queue
import random
import shutil
import sys
import tempfile
import time
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import config
from src import utils
from experiments.e10_capacity_360m import calibration as e10_calibration
from experiments.e10_capacity_360m import e10_common as e10
from experiments.e10_capacity_360m import run as e10_run
from experiments.e8b_readout_generation import run as e8b_run
from experiments.e8b_readout_generation import training as e8b_training


def _must_raise(expected, operation, contains: str | None = None):
    try:
        operation()
    except expected as error:
        if contains is not None and contains not in str(error):
            raise AssertionError(
                f"expected {contains!r} in {type(error).__name__}: {error}"
            ) from error
        return error
    except BaseException as error:
        raise AssertionError(
            f"expected {expected}, got {type(error).__name__}: {error}"
        ) from error
    raise AssertionError(f"expected {expected} to be raised")


@contextmanager
def _strict_state_restored():
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cuda_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    matmul_precision = torch.get_float32_matmul_precision()
    python_rng = random.getstate()
    numpy_rng = e10_calibration.np.random.get_state()
    torch_rng = torch.random.get_rng_state()
    environment = {
        key: os.environ.get(key)
        for key in ("CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED")
    }
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cuda.matmul.allow_tf32 = cuda_matmul_tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.set_float32_matmul_precision(matmul_precision)
        random.setstate(python_rng)
        e10_calibration.np.random.set_state(numpy_rng)
        torch.random.set_rng_state(torch_rng)
        for key, value in environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _write_ledgers(spend_path: Path, retry_path: Path) -> None:
    e10.atomic_write_json(spend_path, e10._initial_spend_ledger())
    e10.atomic_write_json(retry_path, e10._initial_retry_ledger())


def _valid_grant() -> dict:
    return {
        "schema_version": 1,
        "record_type": "e10_calibration_grant",
        "task_id": e10.TASK_ID,
        "status": "GRANTED_ONCE",
        "token": e10.CALIBRATION_TOKEN,
        "utc": e10.utc_now(),
        "binding": e10.binding_record(),
        "allowed_arm": "B4r",
        "identity": "random_smollm2_360m",
        "maximum_gpu_hours": e10.CALIBRATION_MAX_GPU_HOURS,
        "fixed_segments": e10.calibration_grant_segments(),
        "scientific_cell_authorized": False,
    }


@contextmanager
def _temporary_shared_paths(root: Path):
    """Redirect every shared E10 coordination path into one test directory."""
    paths = {
        "SPEND_LEDGER": root / "spend.json",
        "RETRY_LEDGER": root / "retry.json",
        "GOVERNANCE_LOCK": root / "governance.lock",
        "INITIALIZATION_PENDING": root / "initialization-pending.json",
        "CALIBRATION_CLAIM_PATH": root / "calibration-claim.json",
        "CALIBRATION_COMPLETION_PATH": root / "calibration-completion.json",
        "LEDGER_FAILURE_PATH": root / "ledger-failure.json",
        "CALIBRATION_GRANT_PATH": root / "calibration-grant.json",
        "CALIBRATION_REVOCATION_PATH": root / "calibration-revocation.json",
        "CALIBRATION_PATH": root / "calibration.json",
        "GATE1_PATH": root / "gate1.json",
        "RETRY_AUTHORIZATION_DIR": root / "retry-authorizations",
        "CORE_AUTHORIZATION_DIR": root / "core-authorization",
    }
    with ExitStack() as stack:
        for name, path in paths.items():
            stack.enter_context(mock.patch.object(e10, name, path))
        yield paths


def mirror_governance(output: Path) -> Path:
    """Mirror the immutable governance tree into a temporary OUT_DIR.

    The provenance chain the authorization grant binds resolves its preserved
    records through OUT_DIR, so a test that redirects OUT_DIR has to carry those
    records with it. The copy is byte-identical and the real tree is untouched.
    """
    output = Path(output)
    shutil.copytree(e10.OUT_DIR, output, dirs_exist_ok=True)
    return output


# The real shared-state authorization path, captured before any test patches
# it, so a test that forgets to redirect CORE_AUTHORIZATION_DIR fails loudly
# instead of writing a real scientific authorization.
_REAL_CORE_AUTHORIZATION_DIR = e10.CORE_AUTHORIZATION_DIR


def write_core_authorization_grant(**overrides) -> Path:
    """Fabricate the external scientific core grant a test needs.

    Only a test may do this, and only inside a temporary directory that
    _temporary_shared_paths has redirected CORE_AUTHORIZATION_DIR into. No
    project source writes such a record, which
    phase3.assert_no_self_granted_authorization proves by scanning the sources.
    """
    if e10.CORE_AUTHORIZATION_DIR == _REAL_CORE_AUTHORIZATION_DIR:
        raise AssertionError(
            "a test must never write the real E10 core authorization grant; "
            "redirect CORE_AUTHORIZATION_DIR into a temporary directory first"
        )
    body = {
        **e10.core_authorization_grant_expectations(),
        "utc": e10.utc_now(),
        "binding": e10.binding_record(),
    }
    body.update({key: value for key, value in overrides.items()
                 if key != "grant_id"})
    grant = {
        "grant_id": overrides.get("grant_id",
                                  e10.core_authorization_grant_id(body)),
        **body,
    }
    path = e10.core_authorization_grant_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    e10.atomic_write_json(path, grant)
    return path


def _valid_calibration_spend_entry(*, identity="random_smollm2_360m",
                                   arm="B4r", occupancy_ns=3_600_000_000_000):
    body = {
        "identity": identity,
        "arm": arm,
        "context": "phase0_calibration",
        "cell": None,
        "gpu_occupancy_ns": occupancy_ns,
        "measurement": "process_monotonic_ns",
        "outcome": "completed",
        "host": "unit-test-host",
        "pid": 1,
        "started_utc": "2026-08-10T00:00:00Z",
        "ended_utc": "2026-08-10T00:00:01Z",
        "source_digest": "1" * 64,
        "config_digest": "2" * 64,
        "protocol_family": e10.PROTOCOL_FAMILY,
        "claim_sha256": "3" * 64,
        "recipe_digests": {"B4r_train_40k_seed0": "4" * 64},
    }
    return {"entry_id": e10.sha256_bytes(e10.canonical_json_bytes(body)), **body}


def write_retry_authorization(root: Path, arm: str, scale: str, seed: int,
                              *, reason: str = "node_failure",
                              projected_hours: float = 1.0,
                              authorized_by: str = "user",
                              automatic: bool = False,
                              nonce: str | None = None) -> dict:
    """Fabricate the out-of-band retry authorization a test needs.

    Only a test may do this. No project source writes such a record, which
    phase1.assert_no_automatic_retry proves by scanning the sources.
    """
    nonce = nonce or e10.sha256_bytes(
        f"{arm}/{scale}/{seed}/{authorized_by}".encode("ascii")
    )
    failure = root / f"failed-{arm}-{scale}-seed{seed}-{nonce[:8]}.json"
    if not failure.exists():
        e10.atomic_write_json(
            failure, {"status": "CELL_FAILED", "cell": [arm, scale, seed]}
        )
    record = {
        "schema_version": 1,
        "record_type": "e10_retry_authorization",
        "task_id": e10.PHASE1_TASK_ID,
        "status": "AUTHORIZED_ONCE",
        "utc": e10.utc_now(),
        "arm": arm,
        "scale": scale,
        "seed": seed,
        "identity": e10.model_identity(arm),
        "reason": reason,
        "authorized_by": authorized_by,
        "authorization_nonce": nonce,
        "automatic": automatic,
        "failed_cell_record": {
            "path": str(failure.relative_to(e10.PROJECT_ROOT)),
            "sha256": e10.sha256_file(failure),
        },
        "projected_hours": projected_hours,
    }
    path = e10.retry_authorization_path(nonce)
    path.parent.mkdir(parents=True, exist_ok=True)
    e10.atomic_write_json(path, record)
    return record


def _resign_permit(permit: e10.CalibrationPermit, **changes):
    candidate = replace(permit, **changes, signature="")
    values = {
        field: getattr(candidate, field)
        for field in candidate.__dataclass_fields__
        if field != "signature"
    }
    signature = hmac.new(
        e10._PERMIT_SIGNING_SECRET,
        e10.canonical_json_bytes(values),
        hashlib.sha256,
    ).hexdigest()
    return replace(candidate, signature=signature)


def _immutable_writer_worker(path_text: str, writer: int, start, results) -> None:
    start.wait()
    try:
        e10.atomic_write_json(Path(path_text), {"writer": writer})
    except FileExistsError:
        results.put(("refused", writer, "FileExistsError"))
    except BaseException as error:
        results.put(("error", writer, type(error).__name__, str(error)))
    else:
        results.put(("won", writer))


def _calibration_claim_worker(start, results) -> None:
    start.wait()
    try:
        claim = e10.claim_calibration()
    except (AssertionError, FileExistsError) as error:
        results.put(("refused", type(error).__name__, str(error)))
    except BaseException as error:
        results.put(("error", type(error).__name__, str(error)))
    else:
        results.put(("won", claim.claim_id))


def _collect_concurrent(processes, start, results) -> list[tuple]:
    for process in processes:
        process.start()
    start.set()
    observed = []
    try:
        for _ in processes:
            observed.append(results.get(timeout=20.0))
    except queue.Empty as error:
        raise AssertionError("concurrent E10 worker did not report") from error
    finally:
        deadline = time.monotonic() + 20.0
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        stragglers = [process for process in processes if process.is_alive()]
        for process in stragglers:
            process.terminate()
            process.join(timeout=2.0)
        results.close()
        results.join_thread()
    if stragglers:
        raise AssertionError(f"{len(stragglers)} concurrent E10 workers hung")
    bad_exits = [process.exitcode for process in processes if process.exitcode != 0]
    if bad_exits:
        raise AssertionError(f"concurrent E10 worker exits were {bad_exits}")
    return observed


def test_authorization_isolation_and_exact_arms() -> None:
    assert e8b_run.TRAINING_AUTHORIZED == "core-matrix-approved"
    assert e8b_run.CORE_ARMS == ("B1", "B2", "B3")
    assert e10.CORE_ARMS == ("B4", "B4r")
    assert e10.E10_TRAINING_AUTHORIZED is None
    assert e10.E10_CALIBRATION_AUTHORIZED is None
    assert e8b_run.TRAINING_AUTHORIZED != e10.CORE_AUTHORIZATION_TOKEN
    with (
        mock.patch.object(config, "E10_PER_IDENTITY_CEILING_HOURS", 10.0),
        mock.patch.object(config, "E10_PER_CELL_WALL_CLOCK_HOURS", 2.0),
    ):
        _must_raise(
            SystemExit,
            lambda: e10.assert_core_entry_authorized("B4", "train_40k", 0),
            "scientific authorization",
        )


def test_pair_order_and_size_qualified_identities() -> None:
    expected = [
        (arm, scale, seed)
        for scale in ("train_40k", "train_250k")
        for seed in (0, 1, 2)
        for arm in ("B4", "B4r")
    ]
    assert e10.pair_preserving_order() == expected
    assert e10_run.core_order() == expected
    assert e10.model_identity("B4") == "pretrained_smollm2_360m"
    assert e10.model_identity("B4r") == "random_smollm2_360m"
    _must_raise(AssertionError, lambda: e10.model_identity("B3"))
    _must_raise(
        AssertionError,
        lambda: e10.pair_preserving_order([
            ("B4", "train_40k", 0),
        ]),
        "unpaired",
    )

    malformed = e10._initial_spend_ledger()
    malformed["entries"] = [{"identity": "pretrained_smollm2", "gpu_hours": 0.0}]
    _must_raise(AssertionError, lambda: e10._validate_spend_ledger(malformed))


def test_recipe_hash_pairing_and_e8b_noncollision() -> None:
    assert e10.SCHEDULER_HORIZON_EPOCHS == 100
    e10_full = set()
    e10_pairs = set()
    for scale in e10.CORE_SCALES:
        for seed in e10.CORE_SEEDS:
            left = e10.build_recipe("B4", scale, seed)
            right = e10.build_recipe("B4r", scale, seed)
            assert left["arm"] == "B4" and right["arm"] == "B4r"
            assert left["frozen_model_identity"] != right["frozen_model_identity"]
            assert e10.recipe_sha256(left) != e10.recipe_sha256(right)
            assert e10.paired_recipe_sha256(left) == e10.paired_recipe_sha256(right)
            assert e10.verify_recipe_inheritance("B4", scale, seed)[
                "inherited_fields_identical"
            ]
            assert e10.verify_recipe_inheritance("B4r", scale, seed)[
                "inherited_fields_identical"
            ]
            e10_full.update((e10.recipe_sha256(left), e10.recipe_sha256(right)))
            e10_pairs.add(e10.paired_recipe_sha256(left))
    assert len(e10_full) == 12
    assert len(e10_pairs) == 6

    e8b_full = set()
    e8b_pairs = set()
    for arm in e8b_run.CORE_ARMS:
        for scale in e8b_run.CORE_SCALES:
            for seed in e8b_run.CORE_SEEDS:
                recipe = e8b_training.build_core_recipe(arm, scale, seed)
                e8b_full.add(e8b_training.recipe_sha256(recipe))
                e8b_pairs.add(e8b_training.paired_recipe_sha256(recipe))
    assert e10_full.isdisjoint(e8b_full | e8b_pairs)
    assert e10_pairs.isdisjoint(e8b_full | e8b_pairs)


def test_calibration_scheduler_uses_inherited_horizon() -> None:
    optimizer = object()
    scheduler = object()
    with mock.patch.object(
        e10_calibration.e8b_training,
        "make_optimizer",
        return_value=optimizer,
    ), mock.patch.object(
        e10_calibration.e8b_training,
        "make_scheduler",
        return_value=scheduler,
    ) as make_scheduler:
        recipe, observed_optimizer, observed_scheduler = (
            e10_calibration._make_optimizer_and_scheduler(
                object(), "train_250k"
            )
        )
    assert observed_optimizer is optimizer
    assert observed_scheduler is scheduler
    assert "horizon 100 x steps_per_epoch" in recipe["scheduler"]
    assert make_scheduler.call_args.kwargs["total_steps"] == (
        100 * e10.STEPS_PER_EPOCH["train_250k"]
    )

def test_retry_policy_exact_fit_and_second_retry_refusal() -> None:
    assert e10.PERMITTED_RETRY_REASONS == {
        "node_failure",
        "corrupted_checkpoint",
        "gate_requires_corrected_rerun",
    }
    assert e10.FORBIDDEN_RETRY_REASONS == {
        "low_accuracy",
        "unfavourable_contrast",
        "seed_disagreement",
        "loss_curve_shape",
        "try_another_seed",
    }
    for reason in e10.PERMITTED_RETRY_REASONS:
        assert e10.validate_retry_reason(reason) == reason
    for reason in e10.FORBIDDEN_RETRY_REASONS | {"try another seed"}:
        _must_raise(AssertionError, lambda reason=reason: e10.validate_retry_reason(reason))

    with tempfile.TemporaryDirectory(
        prefix=".e10-retry-test-", dir=e10.PROJECT_ROOT
    ) as directory:
        root = Path(directory)
        with (
            _temporary_shared_paths(root) as paths,
            mock.patch.object(config, "E10_PER_IDENTITY_CEILING_HOURS", 10.0),
            mock.patch.object(config, "E10_PER_CELL_WALL_CLOCK_HOURS", 2.0),
            mock.patch.object(config, "E10_HEADROOM_FLOOR_HOURS", 1.0),
        ):
            _write_ledgers(paths["SPEND_LEDGER"], paths["RETRY_LEDGER"])
            boundary_spend = e10._initial_spend_ledger()
            boundary_spend["entries"] = [
                _valid_calibration_spend_entry(
                    occupancy_ns=7 * 3_600_000_000_000
                )
            ]
            fit = e10.retry_admissible(
                "B4r", 2.0, boundary_spend, e10._initial_retry_ledger(),
                remaining_mandatory_hours=0.0,
            )
            assert fit["projected_total_hours"] == 9.0
            assert fit["admissible"] is True

            over = e10._initial_spend_ledger()
            over["entries"] = [
                _valid_calibration_spend_entry(
                    occupancy_ns=7 * 3_600_000_000_000 + 1
                )
            ]
            assert e10.retry_admissible(
                "B4r", 2.0, over, e10._initial_retry_ledger(),
                remaining_mandatory_hours=0.0,
            )["admissible"] is False
            assert e10.retry_admissible(
                "B4r", 2.000001, e10._initial_spend_ledger(),
                e10._initial_retry_ledger(),
                remaining_mandatory_hours=0.0,
            )["admissible"] is False
            _must_raise(
                AssertionError,
                lambda: e10.retry_admissible(
                    "B4r", 1.0, e10._initial_spend_ledger(),
                    e10._initial_retry_ledger(),
                ),
                "remaining-program reservation",
            )

            evidence = root / "retry-evidence.json"
            e10.atomic_write_json(evidence, {"status": "NODE_FAILURE"})
            e10.atomic_write_json(paths["GATE1_PATH"], {
                "status": "GATE1_READY_TEST",
                "binding": e10.binding_record(),
                "per_cell_projections": {
                    "train_40k": {
                        "measured_times_1_25_stress_hours": 1.0,
                    },
                    "train_250k": {
                        "measured_times_1_25_stress_hours": 2.0,
                    },
                },
            })
            # No automatic retry: the same call that used to succeed now
            # refuses until a fresh out-of-band authorization exists.
            _must_raise(
                AssertionError,
                lambda: e10.record_forced_retry(
                    "B4r", "train_40k", 0, "node_failure",
                    str(evidence.relative_to(e10.PROJECT_ROOT)), 1.0,
                ),
                "no fresh retry authorization",
            )
            write_retry_authorization(root, "B4r", "train_40k", 0)
            recorded = e10.record_forced_retry(
                "B4r", "train_40k", 0, "node_failure",
                str(evidence.relative_to(e10.PROJECT_ROOT)), 1.0,
            )
            assert len(recorded["retries"]) == 1
            assert recorded["retries"][0]["identity"] == "random_smollm2_360m"
            assert recorded["retries"][0]["fit"]["projected_total_hours"] == 9.0
            assert recorded["retries"][0]["authorized_by"] == "user"
            assert e10._validate_retry_ledger(recorded) == recorded
            write_retry_authorization(
                root, "B4r", "train_40k", 0, nonce="c" * 64
            )
            _must_raise(
                AssertionError,
                lambda: e10.record_forced_retry(
                    "B4r", "train_40k", 0, "node_failure",
                    str(evidence.relative_to(e10.PROJECT_ROOT)), 1.0,
                ),
                "second forced retry",
            )


def test_core_refuses_with_unset_constants_before_side_effects() -> None:
    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("side effect reached")

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "never-created"
        with (
            mock.patch.object(config, "E10_PER_IDENTITY_CEILING_HOURS", None),
            mock.patch.object(config, "E10_PER_CELL_WALL_CLOCK_HOURS", None),
            mock.patch.object(e10, "OUT_DIR", output),
            mock.patch.object(e10, "model_snapshot_dir", side_effect=forbidden),
            mock.patch.object(e10, "read_spend_ledger", side_effect=forbidden),
            mock.patch.object(torch.cuda, "is_available", side_effect=forbidden),
        ):
            _must_raise(
                SystemExit,
                lambda: e10_run.core_cell("B4", "train_40k", 0),
                "ceiling is unset",
            )
        assert calls == []
        assert not output.exists()


def test_ledgers_absent_malformed_and_unreadable_fail_closed() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        with _temporary_shared_paths(root) as paths:
            spend_path = paths["SPEND_LEDGER"]
            retry_path = paths["RETRY_LEDGER"]
            _must_raise(AssertionError, e10.read_spend_ledger, "absent")
            _must_raise(AssertionError, e10.read_retry_ledger, "absent")
            _must_raise(
                AssertionError,
                lambda: e10.read_spend_ledger(allow_missing=True),
                "absent",
            )

            spend_path.write_text("{")
            retry_path.write_text("[]")
            _must_raise(AssertionError, e10.read_spend_ledger, "unreadable")
            _must_raise(AssertionError, e10.read_retry_ledger, "root is not a map")

            spend_path.write_text(json.dumps(e10._initial_spend_ledger()))
            with mock.patch.object(Path, "read_text", side_effect=PermissionError("denied")):
                _must_raise(AssertionError, e10.read_spend_ledger, "unreadable")

            valid_spend = e10._initial_spend_ledger()
            assert valid_spend["schema_version"] == 2
            valid_spend["entries"] = [_valid_calibration_spend_entry()]
            assert e10._validate_spend_ledger(valid_spend) == valid_spend
            assert e10.identity_hours(
                "random_smollm2_360m", valid_spend
            ) == 1.0

            extra_key = copy.deepcopy(valid_spend)
            extra_key["unexpected"] = True
            _must_raise(
                AssertionError,
                lambda: e10._validate_spend_ledger(extra_key),
                "schema keys mismatch",
            )
            malformed_spend = e10._initial_spend_ledger()
            malformed_spend["entries"] = [
                _valid_calibration_spend_entry(occupancy_ns=True)
            ]
            _must_raise(
                AssertionError,
                lambda: e10._validate_spend_ledger(malformed_spend),
                "invalid occupancy",
            )
            unknown_identity = e10._initial_spend_ledger()
            unknown_identity["entries"] = [
                _valid_calibration_spend_entry(identity="random_smollm2")
            ]
            _must_raise(
                AssertionError,
                lambda: e10._validate_spend_ledger(unknown_identity),
                "unknown identity",
            )

            valid_retry = e10._initial_retry_ledger()
            assert valid_retry["schema_version"] == 2
            assert e10._validate_retry_ledger(valid_retry) == valid_retry
            retry_extra = copy.deepcopy(valid_retry)
            retry_extra["unexpected"] = True
            _must_raise(
                AssertionError,
                lambda: e10._validate_retry_ledger(retry_extra),
                "schema keys mismatch",
            )
            malformed_retry = e10._initial_retry_ledger()
            malformed_retry["retries"] = [{"identity": "random_smollm2"}]
            _must_raise(
                AssertionError,
                lambda: e10._validate_retry_ledger(malformed_retry),
                "schema keys mismatch",
            )

            e10.atomic_write_json(paths["LEDGER_FAILURE_PATH"], {
                "operation": "unit-test",
            })
            _must_raise(
                AssertionError, e10.read_spend_ledger, "failure latch"
            )
            paths["LEDGER_FAILURE_PATH"].unlink()
            e10.atomic_write_json(paths["INITIALIZATION_PENDING"], {
                "status": "INITIALIZATION_PENDING",
            })
            _must_raise(
                AssertionError, e10.read_retry_ledger, "initialization"
            )


def test_lock_timeout_never_reclaims_and_owner_change_is_preserved() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        stale = root / "stale.lock"
        stale_payload = {"lock_id": "stale-owner", "manual_only": True}
        stale.write_text(json.dumps(stale_payload))

        def contend_for_stale_lock():
            with e10.exclusive_file_lock(stale, timeout_seconds=0.01):
                raise AssertionError("stale E10 lock was reclaimed automatically")

        _must_raise(
            AssertionError, contend_for_stale_lock, "remained held; refusing"
        )
        assert json.loads(stale.read_text()) == stale_payload

        changed = root / "changed-owner.lock"

        def change_owner_while_held():
            with e10.exclusive_file_lock(changed):
                e10.atomic_replace_json(changed, {"lock_id": "other-owner"})

        _must_raise(
            AssertionError, change_owner_while_held, "owner changed"
        )
        assert e10.read_json_mapping(changed)["lock_id"] == "other-owner"

        unreadable = root / "unreadable-owner.lock"

        def corrupt_owner_while_held():
            with e10.exclusive_file_lock(unreadable):
                unreadable.write_text("{")

        _must_raise(
            AssertionError, corrupt_owner_while_held, "became unreadable"
        )
        assert unreadable.exists()


def test_memory_gate_fires_at_and_above_eighty_percent() -> None:
    assert e10.memory_gate(70, 79, 100)["fires"] is False
    at_limit = e10.memory_gate(70, 80, 100)
    assert at_limit["reserved_fraction"] == 0.80
    assert at_limit["fires"] is True
    assert e10.memory_gate(70, 81, 100)["fires"] is True
    _must_raise(AssertionError, lambda: e10.memory_gate(0, 0, 0))


def test_calibration_fixed_bounds_and_dev_selection_digest() -> None:
    assert (
        e10_calibration.CALIBRATION_SEED,
        e10_calibration.S1_WARMUP,
        e10_calibration.S1_MEASURED,
        e10_calibration.S2_WARMUP,
        e10_calibration.S2_MEASURED,
        e10_calibration.S3_ROWS,
        e10_calibration.S3_ROW_SEED,
        e10_calibration.S3_WARMUP_BATCHES,
        e10_calibration.S3_ALLOWANCE,
        e10_calibration.S4_ROWS,
        e10_calibration.BATCH_SIZE,
    ) == (0, 20, 25, 20, 60, 1024, 20260810, 2, 1.066, 6, 128)
    assert 20 + 25 < e10.STEPS_PER_EPOCH["train_40k"]
    assert 20 + 60 < e10.STEPS_PER_EPOCH["train_250k"]
    indices = e10_calibration.selected_dev_indices(7714)
    assert len(indices) == 1024
    assert indices[:10].tolist() == [3, 6, 21, 24, 28, 29, 30, 52, 55, 56]
    assert e10_calibration.selection_sha256(indices) == (
        "75c51e94e8dd161922d9b3b4946f7f456bd24318ed909952b5583bfb8132f533"
    )
    _must_raise(
        AssertionError, lambda: e10_calibration.selected_dev_indices(7713)
    )


def test_timing_dataset_requests_only_ids_and_collates_without_labels() -> None:
    class FakeStores:
        image_row = {"i1": 0, "i2": 1}
        image_tokens = e10_calibration.np.arange(
            12, dtype=e10_calibration.np.float16
        ).reshape(2, 3, 2)
        question_tokens = e10_calibration.np.arange(
            10, dtype=e10_calibration.np.float16
        ).reshape(5, 2)

        @staticmethod
        def question_span(question_id):
            return {"q1": (0, 2), "q2": (2, 3)}[question_id]

    frame = e10_calibration.pd.DataFrame({
        "questionId": ["q1", "q2"],
        "imageId": ["i1", "i2"],
    })
    manifest = e10.V2_DIR / "dev.csv"
    with mock.patch.object(
        e10_calibration.pd, "read_csv", return_value=frame
    ) as reader:
        dataset = e10_calibration.LabelFreeTimingDataset(
            manifest, FakeStores()
        )
    assert reader.call_args.args == (manifest,)
    assert reader.call_args.kwargs["usecols"] == ["questionId", "imageId"]
    assert "label" not in reader.call_args.kwargs["usecols"]
    assert len(dataset) == 2
    images, questions, lengths, mask = e10_calibration.collate_label_free(
        [dataset[0], dataset[1]]
    )
    assert tuple(images.shape) == (2, 3, 2)
    assert tuple(questions.shape) == (2, 3, 2)
    assert lengths.tolist() == [2, 3]
    assert mask.tolist() == [[False, False, True], [False, False, False]]
    unapproved = Path("unapproved").with_suffix(bytes([46, 99, 115, 118]).decode())
    with mock.patch.object(e10_calibration.pd, "read_csv") as reader:
        _must_raise(
            AssertionError,
            lambda: e10_calibration.LabelFreeTimingDataset(
                unapproved, FakeStores()
            ),
            "exact development manifest",
        )
        reader.assert_not_called()
    with mock.patch.object(e10_calibration.tokens_data, "TokenDataset") as dataset:
        _must_raise(
            AssertionError,
            lambda: e10_calibration.make_train_loader(
                "unregistered", FakeStores()
            ),
            "unknown E10 calibration scale",
        )
        dataset.assert_not_called()


def test_strict_determinism_reimposed_after_repository_reseed() -> None:
    with _strict_state_restored(), (
        mock.patch.object(torch.cuda, "is_available", return_value=False)
    ), mock.patch.object(torch.cuda, "manual_seed_all", return_value=None):
        state = e10.enable_strict_determinism()
        assert state["deterministic_algorithms"] is True
        assert state["warn_only"] is False

        utils.set_seed(17)
        assert torch.is_deterministic_algorithms_warn_only_enabled() is True
        _must_raise(AssertionError, e10.assert_strict_determinism, "warn_only")

        restored = e10.reseed_strict(17)
        assert restored["deterministic_algorithms"] is True
        assert restored["warn_only"] is False


def test_provenance_hash_helpers_on_temporary_files() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        first = root / "first.bin"
        second = root / "second.bin"
        third = root / "third.bin"
        data = b"hello\n"
        first.write_bytes(data)
        second.write_bytes(data)
        third.write_bytes(data + b"changed")
        expected_blob = hashlib.sha1(b"blob 6\0" + data).hexdigest()
        assert e10.git_blob_sha1(first) == expected_blob
        assert e10.sha256_file(first) == hashlib.sha256(data).hexdigest()
        assert e10.sha256_bytes(data) == hashlib.sha256(data).hexdigest()
        assert e10.files_byte_identical(first, second) is True
        assert e10.files_byte_identical(first, third) is False


def test_atomic_write_refuses_overwrite_and_cleans_failure() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        target = root / "record.json"
        digest = e10.atomic_write_json(target, {"value": 1})
        original = target.read_bytes()
        assert digest == hashlib.sha256(original).hexdigest()
        _must_raise(FileExistsError, lambda: e10.atomic_write_json(target, {"value": 2}))
        assert target.read_bytes() == original

        failed = root / "failed.json"
        with mock.patch.object(e10.os, "link", side_effect=OSError("injected")):
            _must_raise(OSError, lambda: e10.atomic_write_json(failed, {"value": 3}))
        assert not failed.exists()
        assert list(root.glob(f".{failed.name}.*.tmp")) == []

        failed_replace = root / "failed-replace.json"
        with mock.patch.object(e10.os, "replace", side_effect=OSError("injected")):
            _must_raise(
                OSError,
                lambda: e10.atomic_replace_json(failed_replace, {"value": 4}),
            )
        assert not failed_replace.exists()
        assert list(root.glob(f".{failed_replace.name}.*.tmp")) == []


def test_concurrent_immutable_writer_has_exactly_one_winner() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        target = root / "immutable.json"
        context = multiprocessing.get_context("fork")
        start = context.Event()
        results = context.Queue()
        processes = [
            context.Process(
                target=_immutable_writer_worker,
                args=(str(target), writer, start, results),
            )
            for writer in range(24)
        ]
        observed = _collect_concurrent(processes, start, results)
        winners = [item for item in observed if item[0] == "won"]
        refused = [item for item in observed if item[0] == "refused"]
        errors = [item for item in observed if item[0] == "error"]
        assert len(winners) == 1
        assert len(refused) == 23
        assert errors == []
        assert all(item[2] == "FileExistsError" for item in refused)
        assert e10.read_json_mapping(target) == {"writer": winners[0][1]}
        assert list(root.glob(f".{target.name}.*.tmp")) == []


def test_stale_source_binding_is_refused() -> None:
    record = {"binding": e10.binding_record()}
    assert e10.assert_current_binding(record, "fresh") == record["binding"]
    stale = copy.deepcopy(record)
    stale["binding"]["source_digest"] = "0" * 64
    _must_raise(
        AssertionError,
        lambda: e10.assert_current_binding(stale, "stale test record"),
        "stale source_digest",
    )
    stale_config = copy.deepcopy(record)
    stale_config["binding"]["config_digest"] = "f" * 64
    _must_raise(
        AssertionError,
        lambda: e10.assert_current_binding(stale_config, "stale config record"),
        "stale config_digest",
    )


def test_grant_permit_revocation_and_e8b_token_isolation() -> None:
    class FakeOptimizer:
        def __init__(self):
            self.steps = 0

        def step(self):
            self.steps += 1

    with tempfile.TemporaryDirectory(
        prefix=".e10-permit-test-", dir=e10.PROJECT_ROOT
    ) as directory:
        root = Path(directory)
        with (
            _temporary_shared_paths(root) as paths,
            _strict_state_restored(),
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            mock.patch.object(
                e10, "validate_model_verification", return_value={"status": "VERIFIED"}
            ),
        ):
            _write_ledgers(paths["SPEND_LEDGER"], paths["RETRY_LEDGER"])
            e10.atomic_write_json(paths["CALIBRATION_GRANT_PATH"], _valid_grant())
            e10.enable_strict_determinism()
            assert e10.validate_calibration_grant()["status"] == "GRANTED_ONCE"
            claim = e10.claim_calibration()
            assert claim.claim_sha256 == e10.sha256_file(
                paths["CALIBRATION_CLAIM_PATH"]
            )
            _must_raise(
                AssertionError,
                e10.validate_calibration_grant,
                "already claimed",
            )
            started_ns = 100_000_000_000
            with mock.patch.object(
                e10.time, "monotonic_ns", return_value=started_ns
            ):
                permit = e10.authorize_calibration(claim)
            assert e10.validate_calibration_session(permit)[
                "calibration_claimed"
            ] is True
            assert permit.deadline_monotonic_ns == (
                started_ns + int(
                    (e10.CALIBRATION_MAX_SECONDS - e10.CALIBRATION_SAFETY_SECONDS)
                    * 1e9
                )
            )
            with mock.patch.object(
                e10.time, "monotonic_ns",
                return_value=permit.deadline_monotonic_ns - 1,
            ):
                assert e10.validate_calibration_permit(permit)[
                    "warn_only"
                ] is False
            with mock.patch.object(
                e10.time, "monotonic_ns",
                return_value=permit.deadline_monotonic_ns,
            ):
                _must_raise(
                    TimeoutError,
                    lambda: e10.validate_calibration_permit(permit),
                    "deadline reached",
                )

            optimizer = FakeOptimizer()
            forged = replace(permit, token=e8b_run.TRAINING_AUTHORIZED)
            _must_raise(
                AssertionError,
                lambda: e10.guarded_optimizer_step(optimizer, forged),
                "forged E10 permit",
            )
            wrong = _resign_permit(
                permit, token=e8b_run.TRAINING_AUTHORIZED
            )
            _must_raise(
                AssertionError,
                lambda: e10.guarded_optimizer_step(optimizer, wrong),
                "wrong E10 permit",
            )
            assert optimizer.steps == 0
            _must_raise(
                AssertionError,
                lambda: e10.validate_calibration_permit(
                    _resign_permit(permit, process_pid=os.getpid() + 1),
                ),
                "another process",
            )
            _must_raise(
                AssertionError,
                lambda: e10.validate_calibration_permit(_resign_permit(
                    permit,
                    deadline_monotonic_ns=permit.deadline_monotonic_ns + 1,
                )),
                "deadlines were altered",
            )
            _must_raise(
                AssertionError,
                lambda: e10.authorize_calibration(
                    replace(claim, owner_nonce="00" * 32)
                ),
                "does not own",
            )

            now = e10.utc_now()
            recipe_digests = {
                scale: e10.recipe_sha256(
                    e10.build_recipe("B4r", scale, 0)
                )
                for scale in e10.CORE_SCALES
            }
            charge = e10.charge_gpu_hours(
                "random_smollm2_360m",
                "phase0_calibration",
                0,
                "aborted_pre_gpu",
                claim_sha256=claim.claim_sha256,
                owner_nonce=claim.owner_nonce,
                recipe_digests=recipe_digests,
                started_utc=now,
                ended_utc=now,
            )
            revocation_sha = e10.revoke_calibration(
                "unit-test finalization",
                "HARD_GATE_HALTED",
                claim_sha256=claim.claim_sha256,
                spend_entry_id=charge["entry_id"],
            )
            calibration_record = {
                "schema_version": 1,
                "record_type": "e10_phase0_calibration",
                "task_id": e10.TASK_ID,
                "status": "HARD_GATE_HALTED",
                "NON_SCIENTIFIC": True,
                "binding": e10.binding_record(),
                "shared_claim": {"sha256": claim.claim_sha256},
                "resource_charge": {
                    **charge,
                    "claim_sha256": claim.claim_sha256,
                },
                "authorization_revocation": {
                    "path": str(
                        paths["CALIBRATION_REVOCATION_PATH"].relative_to(
                            e10.PROJECT_ROOT
                        )
                    ),
                    "sha256": revocation_sha,
                    "status": "REVOKED",
                },
            }
            calibration_sha = e10.atomic_write_json(
                paths["CALIBRATION_PATH"], calibration_record
            )
            e10.complete_calibration(
                claim,
                spend_entry_id=charge["entry_id"],
                calibration_sha256=calibration_sha,
                revocation_sha256=revocation_sha,
            )
            _must_raise(
                AssertionError,
                lambda: e10.validate_calibration_permit(permit),
                "revoked",
            )
            assert optimizer.steps == 0
            assert e10.calibration_is_revoked() is True
            assert e10.assert_shared_state_healthy(
                require_calibration_complete=True
            )["calibration_complete"] is True


def test_concurrent_calibration_claim_has_exactly_one_winner() -> None:
    with tempfile.TemporaryDirectory(
        prefix=".e10-claim-test-", dir=e10.PROJECT_ROOT
    ) as directory:
        root = Path(directory)
        with _temporary_shared_paths(root) as paths, mock.patch.object(
            e10, "validate_model_verification", return_value={"status": "VERIFIED"}
        ):
            _write_ledgers(paths["SPEND_LEDGER"], paths["RETRY_LEDGER"])
            e10.atomic_write_json(paths["CALIBRATION_GRANT_PATH"], _valid_grant())
            context = multiprocessing.get_context("fork")
            start = context.Event()
            results = context.Queue()
            processes = [
                context.Process(
                    target=_calibration_claim_worker, args=(start, results)
                )
                for _ in range(12)
            ]
            observed = _collect_concurrent(processes, start, results)
            winners = [item for item in observed if item[0] == "won"]
            refused = [item for item in observed if item[0] == "refused"]
            errors = [item for item in observed if item[0] == "error"]
            assert len(winners) == 1
            assert len(refused) == 11
            assert errors == []
            claim = e10._validated_calibration_claim()
            assert claim["claim_id"] == winners[0][1]
            _must_raise(
                AssertionError,
                e10.assert_shared_state_healthy,
                "claim is unresolved",
            )
            assert list(root.glob(
                f".{paths['CALIBRATION_CLAIM_PATH'].name}.*.tmp"
            )) == []


def test_development_csv_allowlist_refuses_unknown_reference() -> None:
    suffix = bytes([46, 99, 115, 118]).decode()
    with tempfile.TemporaryDirectory(prefix=".e10-scan-", dir=e10.PROJECT_ROOT) as directory:
        root = Path(directory)
        safe = root / "safe.py"
        bad = root / "bad.py"
        safe.write_text("value = 'development-only'\n")
        bad.write_text(f"value = {'unapproved' + suffix!r}\n")
        assert e10.assert_no_embargo_reference([safe])["passed"] is True
        _must_raise(
            AssertionError,
            lambda: e10.assert_no_embargo_reference([bad]),
            "non-allowlisted CSV path reference",
        )


def test_guarded_step_is_the_only_optimizer_step_call() -> None:
    found = []
    source_root = e10.PROJECT_ROOT / "experiments" / "e10_capacity_360m"
    for path in sorted(source_root.glob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for function in (
            node for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ):
            calls = [
                node for node in ast.walk(function)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "step"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "optimizer"
            ]
            if calls:
                found.extend((path.name, function.name) for _ in calls)
    assert found == [("e10_common.py", "guarded_optimizer_step")]


def test_zero_checkpoint_and_metric_output_helper() -> None:
    with mock.patch.object(
        e10, "assert_no_embargo_reference", return_value={"passed": True}
    ):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            e10.atomic_write_json(root / "governance.json", {"status": "PRE_RESULT"})
            assert e10.assert_no_scientific_outputs(root) == {
                "checkpoint_files": 0,
                "forbidden_metric_fields": 0,
                "embargo_source_scan_passed": True,
            }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "nested" / "checkpoint.pt").write_bytes(b"")
            _must_raise(AssertionError, lambda: e10.assert_no_scientific_outputs(root))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            e10.atomic_write_json(root / "metric.json", {"nested": {"accuracy": 0.0}})
            _must_raise(
                AssertionError, lambda: e10.assert_no_scientific_outputs(root)
            )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            e10.atomic_write_json(
                root / "metric.json", {"held_out_correctness_rate": 0.0}
            )
            _must_raise(
                AssertionError, lambda: e10.assert_no_scientific_outputs(root)
            )


TESTS = (
    test_authorization_isolation_and_exact_arms,
    test_pair_order_and_size_qualified_identities,
    test_recipe_hash_pairing_and_e8b_noncollision,
    test_calibration_scheduler_uses_inherited_horizon,
    test_retry_policy_exact_fit_and_second_retry_refusal,
    test_core_refuses_with_unset_constants_before_side_effects,
    test_ledgers_absent_malformed_and_unreadable_fail_closed,
    test_lock_timeout_never_reclaims_and_owner_change_is_preserved,
    test_memory_gate_fires_at_and_above_eighty_percent,
    test_calibration_fixed_bounds_and_dev_selection_digest,
    test_timing_dataset_requests_only_ids_and_collates_without_labels,
    test_strict_determinism_reimposed_after_repository_reseed,
    test_provenance_hash_helpers_on_temporary_files,
    test_atomic_write_refuses_overwrite_and_cleans_failure,
    test_concurrent_immutable_writer_has_exactly_one_winner,
    test_stale_source_binding_is_refused,
    test_grant_permit_revocation_and_e8b_token_isolation,
    test_concurrent_calibration_claim_has_exactly_one_winner,
    test_development_csv_allowlist_refuses_unknown_reference,
    test_guarded_step_is_the_only_optimizer_step_call,
    test_zero_checkpoint_and_metric_output_helper,
)


def run() -> None:
    for test in TESTS:
        test()
    print(f"E10 tests passed ({len(TESTS)} checks)")


if __name__ == "__main__":
    run()
