# geo-ml — context for Claude

This repo is **mineral prospectivity mapping for Western Australia** (Ni/Cu primarily; lithium as secondary narrative). Stack: Python, geospatial (rasterio, geopandas), XGBoost with **positive-unlabeled (PU) bagging**, SHAP, spatial CV.

## Run the pipeline

```bash
pip install -r requirements.txt   # or use uv/venv project-side
python scripts/run_pipeline.py
```

Defaults: Kalgoorlie bbox, nickel + copper. Options: `--region yilgarn|--region kalgoorlie`, `--commodities nickel copper lithium`, `--skip-download` if grids cached under `data/raw/`, `--skip-cv` to go faster.

## Where the real domain spec lives

- **[README.md](./README.md)** — quick start, structure, methodology summary
- **[`.cursor/rules/geo-ml.mdc`](./.cursor/rules/geo-ml.mdc)** — geology targets, datasets (GA WCS + Australian Mineral Deposits GDB), citations, PU rationale, pitfalls, conventions. Prefer this file over reinventing geology/ML framing.

## Data you need locally once

Australian Mineral Deposits gdb under `data/raw/aus_mineral_deposits/.../Mineral_Deposits.gdb` — see README zip URL. Geo grids download automatically from GA WCS unless cached.

## Outputs

Regeneratable under **`output/`** (gitignored): `prospectivity.tif`, PNGs, `interactive_map.html`, `report.html`.

## Remote

`https://github.com/herbyxo/geo-ml`
