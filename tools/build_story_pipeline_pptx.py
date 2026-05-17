#!/usr/bin/env python3
"""
From the Auto Story Pipeline flowchart PNG:
1. Crop regions → presentation/story_pipeline_assets/*.png
2. Build presentation/Auto_Story_Pipeline.pptx (2 slides: asset grid + full diagram)

Uses only stdlib + Pillow. No python-pptx.

Example:
  python3 tools/build_story_pipeline_pptx.py \\
    --source "/path/to/diagram.png"
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

from PIL import Image

# Widescreen 16:9 EMU
SLIDE_CX = 12192000
SLIDE_CY = 6858000

SRC_W = 1024
SRC_H = 561


def crop_regions(src: Path, out_dir: Path) -> None:
    """Bounding boxes (L, U, R, D) in pixels for ~1024×561 source diagram."""
    im = Image.open(src).convert("RGB")
    specs: dict[str, tuple[int, int, int, int]] = {
        "01_input_story_text": (6, 6, 322, 132),
        "02_parser_router": (318, 2, 694, 148),
        "03_single_character_path_blue": (4, 138, 498, 418),
        "04_multi_character_path_green": (508, 138, 1018, 418),
        "05_output_panels_bottom": (4, 422, 924, 557),
        "06_sidebar_notes": (926, 118, 1018, 325),
        "full_diagram": (0, 0, im.width, im.height),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, box in specs.items():
        crop = im.crop(box)
        crop.save(out_dir / f"{name}.png", "PNG")


def _escape_xml(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _rels_root() -> bytes:
    return b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>
"""


def _rels_slide(num_images: int, slide_idx: int) -> bytes:
    lines = []
    for i in range(num_images):
        lines.append(
            f'  <Relationship Id="rId{i+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"'
            f' Target="../media/slide{slide_idx}_image{i+1}.png"/>'
        )
    lines.append(
        f'  <Relationship Id="rId{num_images+1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout"'
        ' Target="../slideLayouts/slideLayout1.xml"/>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        + "\n".join(lines)
        + "\n</Relationships>\n"
    ).encode()


def _rels_presentation_two_slides() -> bytes:
    return b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide2.xml"/>
  <Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>
</Relationships>
"""


def _presentation_two_slides() -> bytes:
    return b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
  xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" embedTrueTypeFonts="1">
  <p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>
  <p:sldIdLst><p:sldId id="256" r:id="rId2"/><p:sldId id="257" r:id="rId3"/></p:sldIdLst>
  <p:sldSz cx="12192000" cy="6858000" type="screen16x9"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>
"""


def _slide_layout1() -> bytes:
    return b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
  xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSp><p:cNvPr id="1" name=""/><p:cNvGrpSp/><p:nvPr/></p:nvGrpSp><p:grpSpPr/></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sldLayout>
"""


def _slide_master1() -> bytes:
    return b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
  xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
  xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:bg/><p:spTree>
    <p:nvGrpSp><p:cNvPr id="1" name=""/><p:cNvGrpSp/><p:nvPr/></p:nvGrpSp><p:grpSpPr/>
  </p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
</p:sldMaster>
"""


def _slide_master_rels() -> bytes:
    return b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>
"""


def _slide_layout_rels() -> bytes:
    return b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>
"""


def _theme_minimal() -> bytes:
    return b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Office">
<a:themeElements>
<a:clrScheme name="Office"><a:dk1><a:srgbClr val="000000"/></a:dk1><a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
<a:accent1><a:srgbClr val="5B9BD5"/></a:accent1><a:accent2><a:srgbClr val="70AD47"/></a:accent2></a:clrScheme>
<a:fontScheme name="Office"><a:majorFont/><a:minorFont/></a:fontScheme>
<a:fmtScheme name="Office"><a:fillStyleLst/><a:lnStyleLst/><a:effectStyleLst/><a:bgFillStyleLst/></a:fmtScheme>
</a:themeElements></a:theme>
"""


def _doc_props() -> tuple[str, str]:
    app = """<?xml version="1.0" encoding="UTF-8"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">
  <Application>Story pipeline builder</Application></Properties>"""
    core = """<?xml version="1.0" encoding="UTF-8"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Auto Story Pipeline</dc:title></cp:coreProperties>"""
    return app, core


def _content_types_slide1_assets(n_slide1_images: int) -> bytes:
    o = [
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/slides/slide1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>',
        '<Override PartName="/ppt/slides/slide2.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
    ]
    for i in range(1, n_slide1_images + 1):
        o.append(
            f'<Override PartName="/ppt/media/slide1_image{i}.png" ContentType="image/png"/>'
        )
    o.append('<Override PartName="/ppt/media/slide2_image1.png" ContentType="image/png"/>')
    body = "\n".join(o)
    xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Default Extension="png" ContentType="image/png"/>
{body}
</Types>
"""
    return xml.encode()


def slide1_xml_with_grid(labeled_assets: list[tuple[str, int, int, int, int]]) -> str:
    """
    labeled_assets: (label_text, x, y, cx, cy) EMU boxes for pictures in order → rId1..rIdN
    Labels drawn above each box.
    """
    margin_top_title = 120000
    title_h = 500000

    parts: list[str] = []

    parts.append(
        f"""<p:sp><p:nvSp><p:cNvPr id="9001" name="Title"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSp><p:spPr>
<a:xfrm><a:off x="180000" y="{margin_top_title}"/><a:ext cx="{SLIDE_CX - 360000}" cy="{title_h}"/></a:xfrm></p:spPr>
<p:txBody><a:bodyPr/><a:p><a:r><a:rPr sz="2400" b="1" lang="en-US"/><a:t>Auto Story Pipeline — extracted regions</a:t></a:r></a:p></p:txBody></p:sp>"""
    )

    for i, (label, x, y, cx, cy) in enumerate(labeled_assets):
        lid = _escape_xml(label)
        ry = max(180000, y - 340000)
        parts.append(
            f"""<p:sp><p:nvSp><p:cNvPr id="{9100+i}" name="Lbl{i}"/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSp><p:spPr>
<a:xfrm><a:off x="{x}" y="{ry}"/><a:ext cx="{cx}" cy="320000"/></a:xfrm></p:spPr>
<p:txBody><a:bodyPr/><a:p><a:r><a:rPr sz="880" lang="en-US"/><a:t>{lid}</a:t></a:r></a:p></p:txBody></p:sp>"""
        )

        rid = i + 1
        parts.append(
            f"""<p:pic><p:nvPicPr><p:cNvPr id="{9200+i}" name="Pic{rid}"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>
<p:blipFill><a:blip r:embed="rId{rid}"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
<p:spPr><a:xfrm><a:off x="{x}" y="{y}"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm></p:spPr></p:pic>"""
        )

    inner = "\n".join(parts)
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:cSld><p:spTree>
<p:nvGrpSp><p:cNvPr id="1" name=""/><p:cNvGrpSp/><p:nvPr/></p:nvGrpSp><p:grpSpPr/>
{inner}
</p:spTree></p:cSld></p:sld>"""


def slide2_full_bleed_xml() -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"
xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
<p:cSld><p:spTree>
<p:nvGrpSp><p:cNvPr id="1" name=""/><p:cNvGrpSp/><p:nvPr/></p:nvGrpSp><p:grpSpPr/>
<p:pic><p:nvPicPr><p:cNvPr id="2" name="FullDiagram"/><p:cNvPicPr/><p:nvPr/></p:nvPicPr>
<p:blipFill><a:blip r:embed="rId1"/><a:stretch><a:fillRect/></a:stretch></p:blipFill>
<p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{SLIDE_CX}" cy="{SLIDE_CY}"/></a:xfrm></p:spPr>
</p:pic></p:spTree></p:cSld></p:sld>"""


def build_two_slide_deck(asset_rows: list[tuple[Path, str]], full_png: Path, outp: Path) -> None:
    n = len(asset_rows)
    if n == 0:
        raise ValueError("No assets")

    margin = 120000
    gap_x = 100000
    gap_y = 420000

    usable_w = SLIDE_CX - 2 * margin
    usable_h = SLIDE_CY - margin - gap_y - 550000

    cols = 3 if n >= 3 else n
    rows = (n + cols - 1) // cols
    cell_w = (usable_w - (cols - 1) * gap_x) // cols
    cell_h = (usable_h - (rows - 1) * gap_y) // rows if rows > 1 else usable_h // max(1, rows)

    start_y = 550000
    placements: list[tuple[str, int, int, int, int]] = []
    for i, (_path, lbl) in enumerate(asset_rows):
        row, col = divmod(i, cols)
        x = margin + col * (cell_w + gap_x)
        y = start_y + row * (cell_h + gap_y)
        placements.append((lbl, x, y, cell_w, cell_h))

    s1 = slide1_xml_with_grid(placements)
    s2 = slide2_full_bleed_xml()

    app, core = _doc_props()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", _content_types_slide1_assets(n))
        z.writestr("_rels/.rels", _rels_root())
        z.writestr("docProps/app.xml", app)
        z.writestr("docProps/core.xml", core)
        z.writestr("ppt/presentation.xml", _presentation_two_slides())
        z.writestr("ppt/_rels/presentation.xml.rels", _rels_presentation_two_slides())

        z.writestr("ppt/slides/slide1.xml", s1.encode("utf-8"))
        z.writestr("ppt/slides/_rels/slide1.xml.rels", _rels_slide(n, 1))

        z.writestr("ppt/slides/slide2.xml", s2.encode("utf-8"))
        z.writestr("ppt/slides/_rels/slide2.xml.rels", _rels_slide(1, 2))

        z.writestr("ppt/slideLayouts/slideLayout1.xml", _slide_layout1())
        z.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", _slide_layout_rels())
        z.writestr("ppt/slideMasters/slideMaster1.xml", _slide_master1())
        z.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", _slide_master_rels())
        z.writestr("ppt/theme/theme1.xml", _theme_minimal())

        for i, (fpath, _lbl) in enumerate(asset_rows, start=1):
            z.write(str(fpath), f"ppt/media/slide1_image{i}.png")
        z.write(str(full_png), "ppt/media/slide2_image1.png")

    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_bytes(buf.getvalue())


def resolve_default_source() -> Path:
    repo = Path(__file__).resolve().parents[1]
    candidates = [
        repo.parent / ".cursor" / "projects" / "home-xyz-Desktop-xluo-StoryDiffusion-dsaa2012-proj2-story" / "assets"
        / "c__Users_14483_AppData_Roaming_Cursor_User_workspaceStorage_3f7a93df7d6f4f4d9a59744fc59da748_images_ae0c82adb33608c881f8ac8ac6ffa044-18c778dd-645b-4c62-84ec-46f0a26c264e.png",
        Path(
            "/home/xyz/.cursor/projects/home-xyz-Desktop-xluo-StoryDiffusion-dsaa2012-proj2-story/assets/"
            "c__Users_14483_AppData_Roaming_Cursor_User_workspaceStorage_3f7a93df7d6f4f4d9a59744fc59da748_images_ae0c82adb33608c881f8ac8ac6ffa044-18c778dd-645b-4c62-84ec-46f0a26c264e.png"
        ),
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[-1]


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=None, help="Flowchart PNG (default: auto-discover)")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo / "presentation" / "story_pipeline_assets",
    )
    parser.add_argument(
        "--pptx-out",
        type=Path,
        default=repo / "presentation" / "Auto_Story_Pipeline.pptx",
    )
    args = parser.parse_args()
    src = args.source or resolve_default_source()
    if not src.exists():
        raise SystemExit(f"Source diagram not found: {src}\nPass --source /path/to/diagram.png")

    crop_regions(src, args.out_dir)
    asset_dir = args.out_dir

    rows = [
        (asset_dir / "01_input_story_text.png", "Input · scene-marked story text"),
        (asset_dir / "02_parser_router.png", "Parser & entity-count router"),
        (asset_dir / "03_single_character_path_blue.png", "Single path · Anchor + SDXL-Turbo + IP-Adapter λ=0.3"),
        (asset_dir / "04_multi_character_path_green.png", "Multi path · StoryDiffusion (no single-anchor IP-Adapter)"),
        (asset_dir / "05_output_panels_bottom.png", "Output · N story panels"),
        (asset_dir / "06_sidebar_notes.png", "Sidebar · Future work · DiT (exploratory)"),
    ]

    missing = [p for p, _ in rows if not p.exists()]
    if missing:
        raise SystemExit(f"Missing crops: {missing}")

    full = asset_dir / "full_diagram.png"
    if not full.exists():
        raise SystemExit(f"Missing {full}")

    build_two_slide_deck(rows, full, args.pptx_out)
    print("Assets:", asset_dir)
    print("Deck:  ", args.pptx_out)


if __name__ == "__main__":
    main()
