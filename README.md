# SDG 11 Urban Sustainability Dashboard

![Python](https://img.shields.io/badge/python-3.x-blue.svg)
![pandas](https://img.shields.io/badge/pandas-data--pipeline-150458.svg)
![Tableau](https://img.shields.io/badge/Tableau-dashboard-E97627.svg)

Analyzing air pollution, urbanization, and economic development across 191 countries (2010–2019) through the lens of UN Sustainable Development Goal 11 — Sustainable Cities and Communities.

## Overview

This project combines UN SDG indicator data with World Bank development indicators to explore how urban air quality (PM2.5) relates to economic growth and urbanization at a regional and country level. A Python pipeline cleans, merges, and enriches the raw data; a Tableau workbook turns the result into an interactive dashboard plus a set of supporting analysis views.

**What's inside:**
- A reproducible data-cleaning pipeline (`cleaned_data.py`)
- A merged, analysis-ready dataset (`sdg_dataset_cleaned.csv`)
- An interactive Tableau dashboard (`DV_Group_Assignment_Tableau.twbx`)

## Data Sources

| Source | What it provides | Used for |
|---|---|---|
| [UN SDG Global Database](https://unstats.un.org/sdgs/dataportal) | SDG 11 (Sustainable Cities and Communities) indicator values by country and year | Core indicator values, goal/target/series metadata |
| [World Bank World Development Indicators (WDI)](https://databank.worldbank.org/source/world-development-indicators) | Country-level economic and urban development indicators | GDP per capita, urban population, population density |

WDI indicators used:

| Code | Name |
|---|---|
| `SP.URB.GROW` | Urban population growth (annual %) |
| `SP.URB.TOTL` | Urban population, total |
| `EN.POP.DNST` | Population density (people per sq. km) |
| `NY.GDP.PCAP.CD` | GDP per capita (current US$) |

> **Note:** the raw source files (`sdg_dataset.csv` and the WDI export) aren't included in this repo — only the cleaned/merged output is. To re-run the pipeline from scratch, download fresh copies from the sources above (see [Reproducing the Pipeline](#reproducing-the-pipeline)).

## Repository Structure

```
.
├── cleaned_data.py                    # Cleaning, merging, and feature-engineering pipeline
├── sdg_dataset_cleaned.csv            # Final merged & cleaned dataset (pipeline output)
├── DV_Group_Assignment_Tableau.twbx   # Tableau packaged workbook (dashboard + analysis sheets)
└── README.md
```

## The Data Pipeline

`cleaned_data.py` takes the raw SDG and WDI exports and produces the analysis-ready dataset in a few stages:

1. **Load** — reads the raw SDG CSV and WDI CSV (`latin1` encoding, malformed lines skipped).
2. **Clean SDG data** — drops rows missing country/ISO2/value, removes exact duplicates, parses the nested `Dimensions`/`Attributes` fields into flat `Location`, `Nature`, and `Reporting_Type` columns, and coerces `Year`, `Value`, and count fields to numeric types.
3. **Feature engineering** — adds `Value_Log` (log-transformed value), `Value_Normalized` (min-max normalized within each indicator series), and a `Performance_Flag` (High / Medium / Low, split at 0.66 / 0.33).
4. **Region mapping** — derives a `Region` column (Africa, Americas, Asia, Europe, Oceania) from each country's ISO2 code.
5. **Theme classification** — maps the SDG goal to one of the UN's five "Ps": People, Planet, Prosperity, Peace, Partnerships.
6. **Clean WDI data** — filters to the four indicators above, reshapes from wide (one column per year) to long format, and drops rows with no value.
7. **Merge** — left-joins the cleaned SDG data to WDI data on `ISO2` + `Year` (via a full ISO3→ISO2 lookup table), and flags any negative values for indicators that shouldn't be negative (e.g. population density).
8. **Export** — writes the merged dataset (`sdg_dataset_cleaned.csv`), plus a country-by-year wide pivot (`sdg_wide_pivot.csv`) and a country/year/indicator aggregate (`sdg_aggregated.csv`) — add these to the repo too if you want them versioned.

### Data dictionary

| Column(s) | Description |
|---|---|
| `Goal`, `Goal Short Title EN`, `Target`, `Indicator` | SDG goal/target/indicator identifiers and labels |
| `Country Title EN`, `ISO2`, `UN M49 Code` | Country identifiers |
| `Year`, `Value` | Reporting year and the indicator's value |
| `Series`, `Series Description`, `Series Count` | The specific SDG data series, its description, and source count |
| `Source`, `Foot Notes`, `UUID` | Original metadata from the SDG database |
| `Location`, `Nature`, `Reporting_Type` | Parsed out of the raw `Dimensions`/`Attributes` fields (e.g. urban/rural/national coverage) |
| `Region` *(derived)* | Africa / Americas / Asia / Europe / Oceania |
| `Theme` *(derived)* | People / Planet / Prosperity / Peace / Partnerships |
| `Value_Log`, `Value_Normalized` *(derived)* | Log-transformed and per-series normalized value |
| `Performance_Flag` *(derived)* | High / Medium / Low, based on `Value_Normalized` |
| `Country_Name`, `Country_Code`, `Indicator_Name`, `Indicator_Code`, `WDI_Value` | Merged-in World Bank WDI fields |

## Dashboard

The workbook's main view, **Dashboard 1**, is interactive — filterable by **Year** and **Region** — and combines five visualizations:

- Average air pollution (PM2.5) by region, shown on a map
- Average SDG performance by region
- Top 10 most polluted countries
- Top 10 least polluted countries
- Year-over-year % change by region

The workbook also includes standalone analysis sheets not on the main dashboard:

- Comparative heatmap of selected countries (2010–2019)
- Relationship between economic growth and social development over time
- Trend of average air pollution (PM2.5) by region (2010–2019)
- Year-wise distribution of SDG indicator values (2010–2019)

<!-- TODO: add a screenshot or GIF of Dashboard 1 here, and/or a link if you publish it to Tableau Public -->

## Reproducing the Pipeline

**Requirements:** Python 3.x, plus:

```bash
pip install pandas numpy
```

**Steps:**

1. Clone this repo:
   ```bash
   git clone https://github.com/<your-username>/<repo-name>.git
   cd <repo-name>
   ```
2. Add the two raw source files to the project folder (see [Data Sources](#data-sources) — not included in this repo):
   - `sdg_dataset.csv`
   - `70ab38c5-4a8c-4870-a04b-3b459329775f_Data.csv` (WDI export)
3. Run the pipeline:
   ```bash
   python cleaned_data.py
   ```
   This regenerates `sdg_dataset_cleaned.csv`, `sdg_wide_pivot.csv`, and `sdg_aggregated.csv`.

## Viewing the Dashboard

Open `DV_Group_Assignment_Tableau.twbx` in [Tableau Desktop](https://www.tableau.com/products/desktop) or the free [Tableau Public](https://public.tableau.com/). The packaged workbook already bundles the data extract, so no extra setup is needed — just open and explore.

## Notes & Limitations

- The SDG–WDI merge is a left join, so not every SDG row has a matching WDI value; unmatched rows are kept rather than dropped.
- `Region` comes from a manually maintained ISO2 lookup table; any code not in the table falls back to `"Unknown"`.
- `Performance_Flag` thresholds (0.33 / 0.66) are simple normalized-value cutoffs, not a statistically derived benchmark.

## Key Findings

<!-- TODO: summarize 2-4 takeaways from the dashboard — e.g. which regions had the highest/lowest air pollution, how it trended 2010-2019, and how it related to GDP per capita or urbanization -->

## Team

<!-- TODO: list group members -->

## Course

<!-- TODO: course name, instructor, institution, term -->
