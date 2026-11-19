# OpenMobility 🚦

**Open-source, CPU-first urban mobility intelligence for Paris.**

OpenMobility is a Python-based urban mobility data engineering and analytics project focused on understanding traffic patterns across Paris.

The project combines public traffic measurements, data quality assessment, exploratory analysis, and geographic information to build an extensible foundation for urban mobility intelligence.

Starting with permanent road traffic counters published by the City of Paris, OpenMobility provides a reproducible pipeline to collect, process, analyze, and visualize traffic data.

The long-term objective is to extend this foundation toward traffic forecasting, anomaly detection, road-network analysis, and explainable mobility intelligence.

![Paris hourly traffic monitoring](images/road_analysis.png)

Some prediction forecasting done by a simple XGboost model on 4 days of data:
![Distribution of residuals of predictions](images/residual_distribution.png)
![Scatter plot of residuals vs predictions](images/residuals_vs_prediciton.png)
![Forecast plot of a single sensor in Paris](images/sensor_164_forecast.png)


---

## Features

### 1. Data ingestion
- Retrieve traffic measurements from the Paris Open Data API.
- Paginated downloads with configurable record limits.
- Store raw records in JSON Lines (JSONL) format.
- Save download metadata for reproducibility.

### 2. Data inspection and preprocessing
- Inspect source schemas, inferred data types, and sample records.
- Identify relevant source columns using configurable candidate field names.
- Normalize traffic measurements into a consistent analytical schema.
- Convert timestamps to timezone-aware datetime values.
- Store normalized datasets in compressed Parquet format.

### 3. Data quality assessment
- Calculate missing-value counts and percentages.
- Identify duplicate sensor/timestamp combinations.
- Assess sensor coverage.
- Calculate descriptive statistics for traffic flow and occupancy.
- Examine traffic-state distributions.

### 4. Exploratory data analysis
- Analyze traffic flow by hour of day.
- Compare traffic patterns across weekdays.
- Evaluate missing measurements by sensor.
- Calculate hourly sensor measurement availability.
- Generate analytical CSV outputs and visualizations.

### 5. Geographic integration
- Retrieve the Paris road-network geographic reference.
- Associate traffic sensors with geographic road segments.
- Aggregate measurements at sensor level.
- Calculate geographic matching statistics.
- Export the resulting geographic dataset to Parquet.

### 6. Interactive traffic visualization
- Generate interactive maps using Folium.
- Visualize road segments according to average traffic flow.
- Display traffic statistics through interactive tooltips.
- Use CARTO basemaps with API-key authentication.
- Export standalone HTML maps.

---

## Technology stack

| Technology | Purpose |
|---|---|
| Python 3.11+ | Core development |
| Polars | Data processing and transformations |
| PyArrow | Parquet interoperability |
| GeoPandas | Geographic data processing |
| Shapely | Geometric operations |
| Folium | Interactive geographic visualization |
| Matplotlib | Exploratory charts |
| NumPy | Numerical operations |
| Typer | Command-line interface |
| Rich | Terminal output |
| Requests | API communication |

The project follows a CPU-first approach and does not require a GPU for its current data ingestion, processing, or visualization workflows.

---

## Data sources

### Paris permanent road traffic counters

**Dataset:** Comptage routier – Données trafic issues des capteurs permanents

- Publisher: Ville de Paris, Direction de la Voirie et des Déplacements.
- Dataset identifier: `comptages-routiers-permanents`.
- Temporal resolution: Hourly.
- Coverage: Rolling 13-month dataset.
- License: Open Database License (ODbL).

**Resources:**

- [Paris Open Data dataset](https://opendata.paris.fr/explore/dataset/comptages-routiers-permanents/)
- [Paris Open Data API](https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/comptages-routiers-permanents/records)

The dataset provides measurements from permanent traffic sensors, including:

- Traffic flow (vehicles per hour).
- Occupancy rate (percentage of the measurement interval during which vehicles are present).
- Traffic-state information, when available.

Traffic flow and occupancy should be interpreted together when assessing congestion.

### Geographic reference

The project also uses the Paris road-network geographic reference:

- Dataset identifier: `referentiel-comptages-routiers`.
- Source: Paris Open Data.
- Format: GeoJSON.
- Geographic reference field: `iu_ac`.

The geographic reference is used to associate traffic measurements with road geometries.

For additional information about data provenance and licensing, see [docs/data_sources.md](docs/data_sources.md).

---

## Installation

### Requirements

- Python 3.11 or newer.
- Git.
- Internet connection for downloading public datasets.

### 1. Clone the repository

```bash
git clone https://github.com/spenam/OpenMobility.git
cd OpenMobility
```

### 2. Create a virtual environment

**Windows PowerShell:**

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

Upgrade pip:

```bash
python -m pip install --upgrade pip
```

Install the project with development and geographic dependencies:

```bash
pip install -e ".[dev,geo]"
```

This installs the core dependencies, testing and development tools, and geographic visualization libraries.

---

## Usage

OpenMobility provides a command-line interface for the complete data workflow.

### Step 1: Download traffic data

Retrieve a sample of 5,000 records:

```bash
openmobility fetch
```

Specify a custom download limit:

```bash
openmobility fetch --max-records 20000 --page-size 100
```

Raw records are stored in:

```text
data/raw/paris_traffic.jsonl
```

Download metadata is also generated alongside the raw dataset.

### Step 2: Inspect the source data

```bash
openmobility inspect
```

This command displays:
- Number of records.
- Available source columns.
- Inferred data types.
- Sample records.

Schema inspection is important because the Paris Open Data API may evolve over time.

### Step 3: Preprocess the data

```bash
openmobility preprocess
```

The preprocessing pipeline:
- Identifies relevant timestamp and sensor columns.
- Detects available traffic measurements.
- Normalizes the source fields.
- Converts timestamps into timezone-aware UTC values.
- Removes records without valid timestamps or sensor identifiers.
- Saves the normalized dataset in Parquet format.

Output:

```text
data/processed/paris_traffic_normalized.parquet
```

If a source field cannot be confidently identified, the pipeline reports the available columns so the candidate mappings can be updated.

### Step 4: Assess data quality

```bash
openmobility quality
```

Generates a data-quality report containing:
- Total observations.
- Number of unique sensors.
- Temporal coverage.
- Duplicate sensor/timestamp combinations.
- Missing-value statistics.
- Traffic measurement statistics.
- Traffic-state distribution.

### Step 5: Perform exploratory analysis

```bash
openmobility explore
```

This command generates exploratory summaries and charts from the normalized dataset.

The analysis includes:

**Traffic flow by hour**

Average and median traffic flow across the 24 hours of the day.

**Traffic flow by weekday**

Comparison of traffic patterns across the seven days of the week.

**Missing measurements by sensor**

Assessment of missing flow and occupancy measurements for individual sensors.

**Hourly sensor availability**

A sensor-hour completeness matrix showing whether both traffic flow and occupancy measurements are available.

Generated outputs:

```text
reports/
└── eda/
    ├── traffic_flow_by_hour.csv
    ├── traffic_flow_by_weekday.csv
    ├── missing_values_by_sensor.csv
    ├── sensor_hour_availability.csv
    └── charts/
        ├── traffic_flow_by_hour.png
        ├── traffic_flow_by_weekday.png
        └── sensor_availability.png
```

A custom output directory can also be specified:

```bash
openmobility explore reports/custom_eda
```

### Step 6: Retrieve geographic reference data

```bash
openmobility fetch-geography
```

Downloads the Paris road-network geographic reference in GeoJSON format.

Output:

```text
data/raw/paris_sensor_geography.geojson
```

### Step 7: Associate traffic data with road geometry

```bash
openmobility map-traffic
```

This command:
1. Reads the geographic reference.
2. Loads normalized traffic measurements.
3. Aggregates traffic statistics by sensor.
4. Matches sensor identifiers with geographic road segments.
5. Calculates geographic matching statistics.
6. Saves the resulting geographic dataset.

Output:

```text
data/processed/paris_sensor_geography.parquet
```

The command also reports the number of matched and unmatched road segments and the matching percentage.

### Step 8: Generate the interactive traffic map

The `map-traffic` command also generates an interactive Folium map after successfully building the geographic dataset.

The map displays:
- Road segments colored according to mean traffic flow.
- Interactive tooltips with sensor and traffic statistics.
- A traffic intensity color scale.
- An interactive layer control.
- A CARTO basemap.

Output:

```text
reports/eda/sensor_traffic_map.html
```

Open the generated HTML file in a web browser to explore the map.

---

## CARTO configuration

OpenMobility supports CARTO basemaps for interactive traffic visualization.

To use CARTO, obtain an API key from [CARTO](https://carto.com/) and configure it as an environment variable.

Create a `.env` file in the project root:

```env
CARTO_API_KEY=your_carto_api_key
```

The application loads this variable when generating the interactive map.

**Security:** Do not commit your `.env` file or expose your API key in public repositories. Since browser-based tile requests require the key to be sent to the client, configure appropriate usage restrictions and quotas in your CARTO account.

---

## Project structure

```text
OpenMobility/
│
├── configs/
│   └── data.yaml
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── docs/
│   └── data_sources.md
│
├── reports/
│   └── eda/
│       ├── charts/
│       └── sensor_traffic_map.html
│
├── src/
│   └── openmobility/
│       │
│       ├── ingestion/
│       │   ├── paris_traffic.py
│       │   └── paris_geography.py
│       │
│       ├── processing/
│       │   ├── normalize.py
│       │   ├── quality.py
│       │   ├── exploratory.py
│       │   └── geography.py
│       │
│       ├── visualization/
│       │   ├── traffic.py
│       │   └── maps.py
│       │
│       └── cli.py
│
├── tests/
├── .github/
│   └── workflows/
├── .gitignore
├── pyproject.toml
└── README.md
```

Generated datasets and reports are stored locally and should not be committed to the repository unless explicitly intended.

---

## Normalized data model

The preprocessing pipeline produces a consistent analytical dataset.

| Column | Description |
|---|---|
| `timestamp` | Time of the observation, normalized to UTC |
| `sensor_id` | Traffic sensor identifier |
| `flow_vehicles_per_hour` | Hourly vehicle count |
| `occupancy_pct` | Occupancy rate |
| `traffic_state` | Traffic-state label, when available |

Not all source records necessarily contain every measurement.

The exploratory analysis converts timestamps to the `Europe/Paris` timezone when examining local traffic patterns.

---

## Engineering principles

OpenMobility follows several design principles:

- **Reproducibility:** Raw source records and download metadata are retained.
- **Data provenance:** Public data sources and licensing are documented.
- **CPU-first development:** Current workflows do not require GPU infrastructure.
- **Columnar processing:** Polars and Parquet are used for efficient analytical workflows.
- **Explicit data validation:** Unrecognized source fields should be investigated rather than silently guessed.
- **Modularity:** Ingestion, processing, geographic integration, and visualization are separated into dedicated modules.
- **Incremental development:** The project starts with reliable data foundations before introducing more advanced models.

---

## Roadmap

The project is being developed incrementally.

### Phase 1 — Data engineering and exploratory analysis

- [x] Paris traffic data ingestion.
- [x] Schema inspection and normalization.
- [x] Parquet-based storage.
- [x] Data-quality reporting.
- [x] Exploratory traffic analysis.
- [x] Sensor availability analysis.
- [x] Geographic reference ingestion.
- [x] Traffic-to-road geographic integration.
- [x] Interactive traffic visualization.

### Phase 2 — Predictive analytics

- [ ] Seasonal-naive forecasting baseline.
- [ ] Statistical and machine-learning forecasting models.
- [ ] Sensor-level traffic prediction.
- [ ] Chronological model evaluation.
- [ ] Anomaly detection.

### Phase 3 — Road-network intelligence

- [ ] Road-network graph construction.
- [ ] Spatial and temporal feature engineering.
- [ ] CPU-feasible graph neural network experiments.
- [ ] Network-level traffic analysis.

### Phase 4 — Mobility intelligence platform

- [ ] FastAPI service.
- [ ] Interactive analytical dashboard.
- [ ] Experiment tracking.
- [ ] Explainable mobility insights.
- [ ] Evidence-grounded mobility intelligence agent.

---

## Limitations and considerations

- Traffic measurements are available only at equipped sensor locations.
- Sensor coverage is not spatially uniform across Paris.
- Geographic matching depends on consistency between sensor identifiers and geographic reference identifiers.
- Missing measurements should be preserved and explicitly considered during analysis.
- Traffic flow alone is insufficient to characterize congestion.
- The current traffic dataset is a rolling 13-month dataset; historical data is available separately.
- Geographic matching statistics should be monitored when the source datasets are updated.

---

## License and attribution

OpenMobility is distributed under the MIT License.

The underlying Paris Open Data datasets have their own licensing requirements, including ODbL obligations.

Users should review the original data-source terms and preserve the required attribution when redistributing datasets or derived products.

See [docs/data_sources.md](docs/data_sources.md) for data provenance and source references.

---

## Contributing

Contributions, suggestions, and improvements are welcome.

Potential areas for contribution include:
- Data ingestion and validation.
- Geographic matching improvements.
- Exploratory traffic analysis.
- Forecasting and anomaly detection.
- Road-network modeling.
- Documentation and testing.

Please open an issue to discuss significant changes before submitting a pull request.

---

**OpenMobility — Building an open foundation for urban mobility intelligence.**