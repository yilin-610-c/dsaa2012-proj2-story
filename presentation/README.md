# Auto Story Pipeline — presentation assets

## Editable diagrams (recommended for your use case)

- **`Auto_Story_Pipeline_editable.pptx`** — Native PowerPoint shapes: rounded rectangles, flowchart diamond, **editable text**, straight **connector** lines (not screenshot crops). Regenerate:

  ```bash
  HTTP_PROXY= HTTPS_PROXY= pip install python-pptx   # if needed
  python3 tools/build_editable_story_pipeline_pptx.py
  ```

- **`Auto_Story_Pipeline_editable.drawio`** — Open at [diagrams.net](https://app.diagrams.net) (Draw.io): every box, arrow, label is draggable and editable in the browser.

## Raster export (PNG slices — not editable as separate objects)

- **`Auto_Story_Pipeline.pptx`** — Slide 1: bitmap crops in a grid. Slide 2: full bleed diagram.
- **`story_pipeline_assets/*.png`** — Cropped PNGs from `tools/build_story_pipeline_pptx.py`

```bash
python3 tools/build_story_pipeline_pptx.py \
  --source "/path/to/your/auto_story_pipeline.png" \
  --out-dir presentation/story_pipeline_assets \
  --pptx-out presentation/Auto_Story_Pipeline.pptx
```

Requires: Pillow (raster crops); Python 3.x.
