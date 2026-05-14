from __future__ import annotations

import importlib.util
import json
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


class FakeImage:
    def __init__(self, label: str) -> None:
        self.label = label

    def save(self, path: Path) -> None:
        path.write_text(self.label, encoding="utf-8")


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


def test_run_generation_saves_identity_images_when_enabled(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()

    class FakeApp:
        @staticmethod
        def process_generation(*args, **kwargs):
            yield [FakeImage("identity"), FakeImage("scene-0"), FakeImage("scene-1")]

    monkeypatch.setattr(module, "validate_config", lambda config: None)
    monkeypatch.setattr(module, "import_gradio_app", lambda root: FakeApp)
    storydiffusion_root = tmp_path / "StoryDiffusion"
    storydiffusion_root.mkdir()
    config = module.ProbeConfig(
        storydiffusion_root=storydiffusion_root,
        output_dir=tmp_path / "outputs",
        sd_type="Unstable",
        use_reference_images=False,
        reference_images=[],
        general_prompt="[Student] human man",
        prompt_array="[Student] full body character reference, human man\n[Student] reading\n[Student] leaving",
        negative_prompt="blurry",
        style="(No style)",
        seed=0,
        num_steps=1,
        guidance_scale=5.0,
        sa32=0.5,
        sa64=0.5,
        id_length=1,
        height=512,
        width=512,
        style_strength_ratio=20.0,
        ip_adapter_strength=0.5,
        comic_type="No typesetting (default)",
        font_choice="Inkfree.ttf",
        character_files="",
        save_image_start_index=1,
        save_identity_images=True,
    )

    saved_paths = module.run_generation(config)

    assert [path.name for path in saved_paths] == ["image_000.png", "image_001.png"]
    assert (tmp_path / "outputs" / "identity_refs" / "identity_000.png").read_text(encoding="utf-8") == "identity"
    assert (tmp_path / "outputs" / "image_000.png").read_text(encoding="utf-8") == "scene-0"
    manifest = json.loads((tmp_path / "outputs" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["identity_images"][0].endswith("identity_refs/identity_000.png")
    assert manifest["identity_image_prompt_map"]["identity_000.png"]["prompt"] == "[Student] full body character reference, human man"
    assert manifest["images"][0].endswith("image_000.png")
