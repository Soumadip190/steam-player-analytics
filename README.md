# Steam Player Analytics & Personalized Recommendation A/B Testing

An end-to-end data science project exploring historical Steam player behaviour, identifying behavioural player segments, and evaluating personalized game recommendations through a simulated A/B testing framework.

## Project Overview

This project uses historical Steam gameplay and purchase records to understand player engagement patterns and explore how personalized game recommendations could influence user interactions.

The project combines:

- Exploratory Data Analysis (EDA)
- Data cleaning and feature engineering
- K-Means player segmentation
- A/B experiment design and simulation
- Statistical hypothesis testing
- Logistic regression and segment-level analysis
- Interactive data visualization with Streamlit and Plotly

## Key Findings

### Dataset Overview

| Metric | Value |
|---|---:|
| Original records | 200,000 |
| Records after deduplication | 199,293 |
| Unique players | 12,393 |
| Unique games | 5,155 |
| Player segments | 3 |

### Player Segmentation

K-Means clustering was used to identify three behavioural groups based on player engagement and game ownership.

| Segment | Players | Median Games Owned | Median Playtime |
|---|---:|---:|---:|
| Casual Players | 6,578 | 1 | 1.5 hours |
| Regular Players | 3,945 | 3 | 92 hours |
| Dedicated Players | 1,870 | 27 | 721.1 hours |

### Simulated A/B Testing Results

The experiment compares two recommendation strategies:

- **Control (A):** Generic recommendations.
- **Treatment (B):** Personalized recommendations.

| Metric | Control | Treatment |
|---|---:|---:|
| Players | 6,265 | 6,128 |
| Clicks | 749 | 992 |
| Conversions | 421 | 547 |
| Click-through rate | 11.96% | 16.19% |
| Conversion rate | 6.72% | 8.93% |

**Overall simulated results:**

- Absolute conversion lift: +2.21 percentage points
- Relative conversion lift: +32.83%
- Click-through rate lift: +4.23 percentage points
- Two-proportion z-test p-value: approximately 0.00000473
- 95% confidence interval for absolute conversion lift: +1.26 to +3.16 percentage points

The simulated results show a statistically significant difference between the two groups.

### Segment-Level Analysis

| Segment | Control Conversion | Treatment Conversion | Absolute Lift |
|---|---:|---:|---:|
| Casual | 5.01% | 7.00% | +1.99 pp |
| Regular | 8.04% | 9.10% | +1.06 pp |
| Dedicated | 10.03% | 15.22% | +5.19 pp |

The segment-level results suggest that the simulated treatment-control differences vary across player groups. However, the overall interaction test did not provide statistically significant evidence of treatment-effect heterogeneity (likelihood-ratio test p-value = 0.14276).

## Methodology

1. **Data Collection:** Loaded historical Steam gameplay and purchase records.
2. **Data Cleaning:** Removed duplicate records and prepared the dataset for analysis.
3. **Exploratory Data Analysis:** Investigated gameplay hours, purchases, player engagement, and game popularity.
4. **Feature Engineering:** Aggregated player-level features, including total interactions, unique games, and total playtime.
5. **Player Segmentation:** Applied log transformation, feature scaling, and K-Means clustering with three clusters.
6. **Experiment Design:** Randomly assigned players to simulated control and treatment groups.
7. **Hypothesis Testing:** Compared conversion rates using two-proportion z-tests and confidence intervals.
8. **Advanced Analysis:** Examined segment-level results, applied Holm correction, and fitted logistic regression models.

## Technology Stack

- **Language:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Statistical Analysis:** SciPy, Statsmodels
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Dashboard:** Streamlit
- **Development:** Google Colab, VS Code
- **Version Control:** Git and GitHub

## Project Structure

```text
steam-player-analytics/
│
├── notebooks/
│   ├── 01_data_collection.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_experiment_design.ipynb
│   ├── 04_hypothesis_testing.ipynb
│   └── 05_advanced_analysis.ipynb
│
├── dashboard/
│   └── app.py
│
├── data/
│   ├── player_features_with_clusters.csv
│   ├── cleaned_steam_data.csv
│   └── simulated_ab_test_data.csv
│
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Running the Dashboard Locally

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd steam-player-analytics
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run Streamlit

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser.

## Important Research Limitations

This project distinguishes historical observational data from simulated experimental outcomes.

- The original Steam dataset contains historical gameplay and purchase behaviour.
- It does not contain actual recommendation exposures, randomized experiment assignments, or observed recommendation conversions.
- The A/B testing outcomes in this project are synthetically generated for analytical demonstration.
- Clicks and conversions were generated independently in the simulation; therefore, the data does not represent a true click-to-purchase funnel.
- The experiment does not establish a real causal effect of personalized recommendations on Steam users.
- The project does not measure actual D1 or D7 retention.

A real-world evaluation would require a properly instrumented randomized experiment with defined exposure, conversion, and retention measurement windows.

## Future Improvements

- Integrate real recommendation exposure and interaction data.
- Build a production-style recommendation system.
- Conduct a properly instrumented randomized A/B test.
- Introduce retention and long-term engagement metrics.
- Explore more advanced player segmentation techniques.
- Deploy the interactive dashboard for public access.


**Data Science Project | 2026**

Built as a practical exploration of player analytics, machine learning, experimentation, and data-driven product decisions.
