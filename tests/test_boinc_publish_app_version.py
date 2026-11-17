# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Publishing app versions: a beta must never share a plan class with a live stable version (it hides the stable one from the scheduler cache)."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "boinc_publish_app_version.py"
_spec = importlib.util.spec_from_file_location("boinc_publish_app_version", _PATH)
assert _spec is not None and _spec.loader is not None
pub: ModuleType = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pub)

SPEC = """<plan_classes>
 <plan_class>
  <name>cuda_hivemind</name>
  <gpu_type>nvidia</gpu_type>
  <cuda/>
  <min_gpu_ram_mb>3500</min_gpu_ram_mb>
  <gpu_peak_flops_scale>0.1</gpu_peak_flops_scale>
 </plan_class>
</plan_classes>
"""


def test_version_xml_marks_only_the_launcher_as_main_program_and_carries_the_plan_class() -> None:
    xml = pub.render_version_xml("app_102_x", ["runtime.tar.xz", "model.tar"], "cuda_hivemind_beta")
    assert xml.count("<main_program/>") == 1 and xml.index("app_102_x") < xml.index("<main_program/>") < xml.index("runtime.tar.xz")
    assert "<plan_class>cuda_hivemind_beta</plan_class>" in xml and xml.count("<file>") == 3


def test_beta_plan_class_is_added_once_copied_from_the_stable_one_and_the_stable_one_is_untouched() -> None:
    out = pub.plan_class_spec_with_beta(SPEC)
    assert out.count("<plan_class>") == 2 and out.endswith("</plan_classes>\n")
    assert "<name>cuda_hivemind_beta</name>" in out and out.count("<gpu_peak_flops_scale>0.2</gpu_peak_flops_scale>") == 1
    assert out.count("<gpu_peak_flops_scale>0.1</gpu_peak_flops_scale>") == 1  # the stable class keeps its scale
    assert out.count("<min_gpu_ram_mb>3500</min_gpu_ram_mb>") == 2 and out.startswith(SPEC.split("</plan_classes>")[0])
    assert pub.plan_class_spec_with_beta(out) == out  # idempotent
    with pytest.raises(SystemExit):
        pub.plan_class_spec_with_beta("<plan_classes>\n</plan_classes>\n")


def test_a_beta_under_a_plan_class_a_stable_version_uses_is_refused() -> None:
    with pytest.raises(SystemExit, match="would get no work"):
        pub.assert_beta_cannot_mask("cuda_hivemind", ["cuda_hivemind"])
    pub.assert_beta_cannot_mask("cuda_hivemind_beta", ["cuda_hivemind"])  # its own class is fine
    pub.assert_beta_cannot_mask("cuda_hivemind", [])  # nothing stable to hide


class FakeStack:
    """Only what publish() touches before it writes anything."""

    dbname = "hmproj"

    def __init__(self, root: Path, stable_classes: str) -> None:
        self.root, self._stable = root, stable_classes
        self.sql_calls: list[str] = []

    def sql(self, statement: str) -> str:
        self.sql_calls.append(statement)
        return self._stable

    def update_versions(self) -> str:  # pragma: no cover - must not be reached
        raise AssertionError("update_versions ran")


def test_publish_stops_before_touching_the_project_when_the_beta_would_mask_the_stable_version(tmp_path: Path) -> None:
    st = FakeStack(tmp_path, "plan_class\ncuda_hivemind\n")
    with pytest.raises(SystemExit, match="refusing"):
        pub.publish(st, tmp_path, 102, "cuda_hivemind", beta=True)  # type: ignore[arg-type]
    assert list(tmp_path.iterdir()) == []  # no apps/ tree, no plan_class_spec.xml, nothing registered
    assert len(st.sql_calls) == 1 and st.sql_calls[0].lstrip().startswith("use hmproj; select")  # it only looked
