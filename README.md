# geo-ml

ML-based mineral prospectivity mapping for Western Australia.

Predicts where nickel-copper and lithium deposits are likely to exist, using publicly available geophysical data and known deposit locations as training labels.

## Quick Start

```bash
# Install dependencies
uv venv && .venv\Scripts\activate
uv pip install -r requirements.txt

# Download deposit database (one time)
python -c "import requests, zipfile, io; r = requests.get('https://data.gov.au/data/dataset/308bd1c4-f495-4d82-b5e8-dabae98dcf69/resource/db7c7c99-d0b1-41de-8dae-30c5051f1306/download/14e96462-b029-469a-9af8-06410f39589b.zip', timeout=120); z = zipfile.ZipFile(io.BytesIO(r.content)); z.extractall('data/raw/aus_mineral_deposits'); print('Done')"

# Run the pipeline
python scripts/run_pipeline.py
```

The pipeline downloads geophysical grids automatically from Geoscience Australia, trains the model, and outputs everything to `output/`.

## What It Does

1. **Downloads** magnetic, gravity, and radiometric grids from Geoscience Australia's Web Coverage Service
2. **Loads** known nickel/copper deposit locations from the Australian Mineral Deposits database (18,719 records)
3. **Engineers 13 features** from the raw grids: TMI, magnetic derivatives (FVD, analytic signal, tilt angle, HGM), Bouguer gravity + gradient, radiometric K/Th/U + ratios (K/Th, Th/U, F-parameter)
4. **Trains** an XGBoost model using Positive-Unlabeled (PU) bagging — the correct approach when you only have confirmed deposits but no confirmed absences
5. **Validates** with spatial cross-validation (latitude-band blocking to prevent spatial leakage)
6. **Outputs** a prospectivity probability map, SHAP feature importance, concentration-area curve, interactive browser map, and an HTML report

## Results (Kalgoorlie Region)

| Metric | Value |
|--------|-------|
| Spatial CV AUC | 0.835 ± 0.055 |
| Top feature | TMI (magnetic intensity) |
| Known deposits in region | 56 |
| Area flagged P > 0.5 | 19.9% |
| Pipeline runtime | ~7s (cached data) |

## CLI Options

```bash
# Default: Kalgoorlie region, nickel + copper
python scripts/run_pipeline.py

# Full Yilgarn Craton
python scripts/run_pipeline.py --region yilgarn

# Different targets
python scripts/run_pipeline.py --commodities nickel copper lithium

# More PU bags (slower, more stable predictions)
python scripts/run_pipeline.py --n-bags 200

# Skip data download (use cached)
python scripts/run_pipeline.py --skip-download

# Skip cross-validation (faster)
python scripts/run_pipeline.py --skip-cv
```

## Output Files

| File | Description |
|------|-------------|
| `output/prospectivity.tif` | GeoTIFF probability map (0-1), loadable in QGIS/ArcGIS |
| `output/prospectivity_map.png` | Static map with deposit overlay |
| `output/shap_importance.png` | SHAP feature importance bar chart |
| `output/concentration_area.png` | C-A curve (% area vs % deposits captured) |
| `output/interactive_map.html` | Interactive browser map with clickable deposits |
| `output/feature_layers.png` | Overview of all input feature grids |
| `output/report.html` | Full standalone HTML report |

## Project Structure

```
geo-ml/
├── geo_ml/                  # Core Python package
│   ├── config.py            # Region/target/parameter configuration
│   ├── ingest/
│   │   ├── ga_geophysics.py # Geoscience Australia WCS download
│   │   └── ozmin.py         # Mineral deposits database loader
│   ├── features/
│   │   └── spatial.py       # Feature engineering (derivatives, ratios)
│   ├── models/
│   │   └── train.py         # PU-bagging XGBoost + SHAP + spatial CV
│   └── viz/
│       └── maps.py          # Plotting, interactive maps, HTML report
├── scripts/
│   └── run_pipeline.py      # End-to-end CLI pipeline
├── notebooks/
│   └── 01_data_exploration.ipynb
├── data/
│   ├── raw/                 # Downloaded grids + deposit DB (gitignored)
│   └── processed/
├── output/                  # Generated maps and reports
└── requirements.txt
```

## Data Sources

- **Geophysical grids**: Geoscience Australia National Compilations (2019), via WCS
- **Deposit locations**: Australian Mineral Deposits database (Geoscience Australia), via data.gov.au
- All data is publicly available under open government licences

## Method

The core challenge is that we have positive labels (known deposits) but no reliable negatives — absence of a known deposit doesn't mean absence of mineralisation. Standard binary classification would treat all unlabeled pixels as "no deposit," contaminating the training set.

**Positive-Unlabeled (PU) bagging** solves this: train N classifiers, each on all positives + a random unlabeled subset as pseudo-negatives, then average predictions. This hedges against accidentally labeling true deposits as negative.

Validation uses **spatial cross-validation** (latitude-band blocking) rather than random splits to prevent spatial autocorrelation from inflating metrics.
