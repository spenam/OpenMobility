# OpenMobility

Open-source, CPU-first urban mobility intelligence for Paris.

OpenMobility starts with public traffic measurements published by the City of Paris. The first milestone is a reproducible ingestion and inspection pipeline; forecasting, anomaly detection and road-network graph learning will build on the resulting data.

## Data source

**Comptage routier – Données trafic issues des capteurs permanents**  
Dataset identifier: `comptages-routiers-permanents`  
Publisher: Ville de Paris, Direction de la Voirie et des Déplacements  
Portal: https://opendata.paris.fr/explore/dataset/comptages-routiers-permanents/  
API: `https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptages-routiers-permanents/records`

The City describes hourly traffic flow and occupancy measurements from permanent sensors, covering more than 3,000 road sections. Occupancy is the percentage of an hourly interval during which vehicles are present; flow is the number of vehicles counted during that interval. The current dataset is a rolling 13-month dataset. Historical data is published separately.

**License:** Open Database License (ODbL). Review attribution and share-alike obligations before redistributing derived databases or combining datasets. See the source portal for current terms.

## Quickstart

Requires Python 3.12 (Python 3.11 should also work) and Git.

```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/openmobility.git
cd openmobility

python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dev]"

# Download a small reproducible sample (5,000 records by default)
openmobility fetch --max-records 5000

# Inspect the returned schema before writing transformation assumptions
openmobility inspect

# Run the initial normalisation (uses detected field names; see notes below)
openmobility preprocess
```

Alternatively, run commands with `python -m openmobility.cli ...` from the repository root.

## Commands

```bash
openmobility fetch --max-records 5000
openmobility fetch --max-records 20000 --page-size 100
openmobility inspect
openmobility preprocess
```

The fetcher saves raw API records as JSON Lines in `data/raw/`. It paginates conservatively and records source metadata. It does not download the entire city dataset by default.

The API schema can evolve. Always run `openmobility inspect` after the first download. The preprocessing step uses candidate field names and reports the available columns if it cannot confidently identify a timestamp, sensor identifier, and traffic measure. Do not silently guess a field mapping.

## Repository layout

```text
openmobility/
├── configs/
│   └── data.yaml
├── data/
│   ├── raw/          # downloaded source records (not committed)
│   ├── interim/      # normalized intermediate files (not committed)
│   └── processed/    # analysis-ready outputs (not committed)
├── docs/
│   └── data_sources.md
├── src/openmobility/
│   ├── cli.py
│   ├── ingestion/
│   │   └── paris_traffic.py
│   └── processing/
│       └── normalize.py
├── tests/
├── .github/workflows/ci.yml
├── pyproject.toml
└── README.md
```

## Data model (target)

The first normalized table aims for one row per sensor and hourly observation:

- `timestamp`: timestamp marking the end of the hourly measurement interval, Europe/Paris local time as published.
- `sensor_id`: source sensor / road-section identifier.
- `flow_vehicles_per_hour`: hourly vehicle count, when supplied.
- `occupancy_pct`: occupancy percentage, when supplied.
- `traffic_state`: source traffic-state label, when supplied.
- `source_record`: original record retained for traceability.

Do not interpret flow alone as congestion: the City notes that flow and occupancy must be considered together. Sensor availability and measurement types differ.

## CPU-first engineering choices

- Polars for columnar data processing.
- DuckDB for local analytical SQL.
- No GPU requirement for ingestion or classical baselines.
- Start with a small sample; scale only after checking API limits, schema and disk usage.
- Later forecasting benchmarks should use chronological splits, not random row splits.

## Roadmap

1. Reproducible Paris traffic ingestion and schema validation.
2. Sensor-level hourly time series and data-quality report.
3. Seasonal-naive and tree-based forecasting baselines.
4. Road-network graph construction and CPU-feasible GNN experiments.
5. Anomaly detection and explainable event summaries.
6. FastAPI service, dashboard, experiment tracking and an evidence-grounded mobility agent.

## Important limitations

This dataset measures traffic at sensor locations / road sections. It is not a complete representation of all movement in Paris, and sensor placement is not uniform. Avoid claiming city-wide traffic conditions from a small sensor subset. Preserve missingness and sensor metadata; do not interpolate across long gaps without documenting the method.
