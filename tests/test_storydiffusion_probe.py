from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import ModuleType


def _load_module() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    path = root / "storydiffusion_gradio_probe" / "run_probe.py"
    spec = importlib.util.spec_from_file_location("run_probe_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeBlocks:
    def launch(self, *args, **kwargs):
        raise AssertionError("Blocks.launch should be patched during import")


class FakeInterface:
    def launch(self, *args, **kwargs):
        raise AssertionError("Interface.launch should be patched during import")


def test_import_gradio_app_patches_and_restores_launch_methods(tmp_path: Path, monkeypatch, capsys) -> None:
    module = _load_module()
    fake_gradio = types.SimpleNamespace(
        blocks=types.SimpleNamespace(Blocks=FakeBlocks),
        Interface=FakeInterface,
    )
    monkeypatch.setitem(sys.modules, "gradio", fake_gradio)
    sys.modules.pop("gradio_app_sdxl_specific_id_low_vram", None)
    app_path = tmp_path / "gradio_app_sdxl_specific_id_low_vram.py"
    app_path.write_text(
        """import gradio
blocks_launch_result = gradio.blocks.Blocks().launch(server_name='0.0.0.0')
interface_launch_result = gradio.Interface().launch()
def process_generation(*args, **kwargs):
    yield []
""",
        encoding="utf-8",
    )
    original_blocks_launch = FakeBlocks.launch
    original_interface_launch = FakeInterface.launch

    try:
        app = module.import_gradio_app(tmp_path)
    finally:
        sys.modules.pop("gradio_app_sdxl_specific_id_low_vram", None)

    assert app.blocks_launch_result is None
    assert app.interface_launch_result is None
    assert FakeBlocks.launch is original_blocks_launch
    assert FakeInterface.launch is original_interface_launch
    assert "[storydiffusion_probe] skipped Gradio launch during import" in capsys.readouterr().out
