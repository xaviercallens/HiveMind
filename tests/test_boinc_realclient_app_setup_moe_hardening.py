from __future__ import annotations

from pathlib import Path
import pytest

from hivemind.boinc.real_client.app import (
    _build,
    boinc_finish,
    class_fingerprint,
    resolve_filename,
    run as run_inner_loop,
)
from hivemind.boinc.real_client.client_setup import (
    Client,
    moe_apps,
    prepare_data_dir,
    project_dir_name,
    toy_app,
)
from hivemind.boinc.real_client.moe_app import (
    build_mixed_domain_train_tokens,
    cosine_segment_lr,
    gpu_gate,
    tokens_sha,
)
from hivemind.boinc.real_client.moe_bridge import (
    _reject,
    _resolve_server_elapsed,
    _spec_flops_bound_infer,
    _spec_flops_bound_train,
)


# ---------------------------------------------------------------------------
# 1. app.py
# ---------------------------------------------------------------------------


def test_real_client_app_resolve_fingerprint_build_and_run_validation(tmp_path: Path) -> None:
    p = resolve_filename("in.json", base=tmp_path)
    assert p == tmp_path / "in.json"
    with pytest.raises(ValueError, match="logical"):
        resolve_filename("", base=tmp_path)
    with pytest.raises(ValueError, match="logical"):
        resolve_filename("../secret.json", base=tmp_path)

    fp = class_fingerprint("cpu", "fp32")
    assert fp["dtype"] == "fp32"
    with pytest.raises(ValueError, match="device"):
        class_fingerprint("tpu")
    with pytest.raises(ValueError, match="dtype"):
        class_fingerprint("cpu", "")

    valid_spec = {
        "wu_id": "wu-1",
        "seed": 42,
        "d_in": 8,
        "d_out": 4,
        "rank": 2,
        "batch": 2,
        "steps": 4,
        "lr": 0.01,
        "spot_check_interval": 2,
    }
    built = _build(valid_spec, "cpu")
    assert len(built) == 5

    with pytest.raises(TypeError, match="spec"):
        _build("bad", "cpu")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="d_in"):
        _build({**valid_spec, "d_in": 0}, "cpu")
    with pytest.raises(ValueError, match="seed"):
        _build({**valid_spec, "seed": True}, "cpu")

    res = run_inner_loop(valid_spec, "cpu", tmp_path / "ckpt.pt")
    assert res["wu_id"] == "wu-1"
    with pytest.raises(ValueError, match="crash_after_step"):
        run_inner_loop(valid_spec, "cpu", tmp_path / "ckpt.pt", crash_after_step=0)
    with pytest.raises(ValueError, match="inject_fault"):
        run_inner_loop(valid_spec, "cpu", tmp_path / "ckpt.pt", inject_fault=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lr"):
        run_inner_loop({**valid_spec, "lr": 0.0}, "cpu", tmp_path / "ckpt.pt")
    with pytest.raises(ValueError, match="spot_check_interval"):
        run_inner_loop({**valid_spec, "spot_check_interval": 0}, "cpu", tmp_path / "ckpt.pt")

    boinc_finish(0, cwd=tmp_path)
    assert (tmp_path / "boinc_finish_called").read_text().strip() == "0"
    with pytest.raises(ValueError, match="status"):
        boinc_finish(True, cwd=tmp_path)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. client_setup.py
# ---------------------------------------------------------------------------


def test_real_client_setup_validation(tmp_path: Path) -> None:
    assert project_dir_name("http://127.0.0.1:8080/hivemind/") == "127.0.0.1_8080_hivemind"
    with pytest.raises(ValueError, match="master_url"):
        project_dir_name(" ")

    assert toy_app("python3")["python"] == "python3"
    with pytest.raises(ValueError, match="python"):
        toy_app("")

    apps = moe_apps("python3", gpu_ram_mb=3600)
    assert len(apps) == 2
    with pytest.raises(ValueError, match="venv_python"):
        moe_apps("")
    with pytest.raises(ValueError, match="gpu_ram_mb"):
        moe_apps("python3", gpu_ram_mb=0)
    with pytest.raises(ValueError, match="gpu_ram_mb"):
        moe_apps("python3", gpu_ram_mb=True)  # type: ignore[arg-type]

    info = prepare_data_dir(tmp_path / "boinc_data", "http://127.0.0.1:8080/", tmp_path)
    assert info["project_dir"].exists()
    with pytest.raises(ValueError, match="master_url"):
        prepare_data_dir(tmp_path / "d2", "", tmp_path)
    with pytest.raises(ValueError, match="gpu_flops"):
        prepare_data_dir(tmp_path / "d2", "http://127.0.0.1/", tmp_path, gpu_flops=0.0)
    with pytest.raises(ValueError, match="install_apps"):
        prepare_data_dir(tmp_path / "d2", "http://127.0.0.1/", tmp_path, install_apps="yes")  # type: ignore[arg-type]

    client = Client(tmp_path / "boinc", tmp_path / "lib", tmp_path / "data", rpc_port=31416)
    assert client.port == 31416
    with pytest.raises(ValueError, match="rpc_port"):
        Client(tmp_path / "boinc", tmp_path / "lib", tmp_path / "data", rpc_port=0)
    with pytest.raises(ValueError, match="rpc_port"):
        Client(tmp_path / "boinc", tmp_path / "lib", tmp_path / "data", rpc_port=70000)
    with pytest.raises(ValueError, match="rpc_port"):
        Client(tmp_path / "boinc", tmp_path / "lib", tmp_path / "data", rpc_port=True)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. moe_app.py
# ---------------------------------------------------------------------------


def test_real_client_moe_app_validation() -> None:
    assert len(tokens_sha([[1, 2, 3]])) == 64
    with pytest.raises(ValueError, match="tokens"):
        tokens_sha("123")  # type: ignore[arg-type]

    mixed = build_mixed_domain_train_tokens(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]],
        {"math": [[11, 12]]},
        target_domain="code",
        replay_ratio=0.2,
        rng_seed=7,
    )
    assert len(mixed) == 5
    with pytest.raises(ValueError, match="target_tokens"):
        build_mixed_domain_train_tokens("bad", None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="target_domain"):
        build_mixed_domain_train_tokens([[1, 2]], None, target_domain="")
    with pytest.raises(ValueError, match="replay_ratio"):
        build_mixed_domain_train_tokens([[1, 2]], None, replay_ratio=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rng_seed"):
        build_mixed_domain_train_tokens([[1, 2]], None, rng_seed=True)  # type: ignore[arg-type]

    lr = cosine_segment_lr(1e-4, 1, 4, round_idx=0, total_rounds=10, min_lr_scale=0.2)
    assert 0.0 < lr <= 1e-4
    with pytest.raises(ValueError, match="base_lr"):
        cosine_segment_lr(0.0, 1, 4)
    with pytest.raises(ValueError, match="segment_idx"):
        cosine_segment_lr(1e-4, -1, 4)
    with pytest.raises(ValueError, match="total_segments"):
        cosine_segment_lr(1e-4, 0, 0)
    with pytest.raises(ValueError, match="min_lr_scale"):
        cosine_segment_lr(1e-4, 0, 4, min_lr_scale=1.5)

    assert gpu_gate({"app": "hivemind_qlora_moe", "min_gpu_ram_mb": 4000}, "train", 8000) is None
    assert gpu_gate({"app": "hivemind_qlora_moe", "min_gpu_ram_mb": 8000}, "train", 4000) is not None
    with pytest.raises(TypeError, match="spec"):
        gpu_gate("bad", "train", 8000)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mode"):
        gpu_gate({}, "", 8000)
    with pytest.raises(ValueError, match="have_mb"):
        gpu_gate({}, "train", -1)


# ---------------------------------------------------------------------------
# 4. moe_bridge.py
# ---------------------------------------------------------------------------


def test_real_client_moe_bridge_validation() -> None:
    v = _reject("wu-1", "host-1", "invalid result")
    assert v.valid is False
    with pytest.raises(ValueError, match="wu"):
        _reject("", "host-1", "invalid")
    with pytest.raises(ValueError, match="host"):
        _reject("wu-1", " ", "invalid")
    with pytest.raises(TypeError, match="detail"):
        _reject("wu-1", "host-1", "invalid", detail="bad")  # type: ignore[arg-type]

    assert _spec_flops_bound_train({"inner_steps": 10, "micro_batch": 2, "seq_len": 64}) > 0.0
    with pytest.raises(TypeError, match="spec"):
        _spec_flops_bound_train("bad")  # type: ignore[arg-type]

    assert _spec_flops_bound_infer({}, [{"token_ids": [1, 2, 3]}]) > 0.0
    with pytest.raises(TypeError, match="spec"):
        _spec_flops_bound_infer("bad", [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="completions"):
        _spec_flops_bound_infer({}, "bad")  # type: ignore[arg-type]

    assert _resolve_server_elapsed({"server_elapsed_seconds": 12.5}, None) == pytest.approx(12.5)
    with pytest.raises(TypeError, match="spec"):
        _resolve_server_elapsed("bad", None)  # type: ignore[arg-type]
