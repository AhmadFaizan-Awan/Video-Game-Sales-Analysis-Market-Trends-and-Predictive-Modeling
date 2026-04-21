# Video Game Sales Dashboard

A polished Streamlit application for exploring video game sales datasets and making sales predictions using machine learning.

## Overview

This project provides an interactive analytics dashboard for the `vgsales.csv` dataset. Users can:

- Explore global and regional sales trends over time
- Compare genres, platforms, and publishers
- Filter results by year, genre, and platform
- Train a Random Forest regression model to predict game sales
- Download filtered data for further analysis

## Key Features

- **Interactive Streamlit dashboard** with custom dark theme styling
- **Market overview visualizations** using Plotly and seaborn
- **Deep dive analysis** for genre, platform, and publisher performance
- **Predictive modeling** with preprocessing and model evaluation
- **Raw data explorer** with filtered CSV export

## Project Structure

- `app.py` — main Streamlit application
- `main.py` — simple project entry placeholder
- `vgsales.csv` — dataset used by the dashboard
- `pyproject.toml` — project metadata and dependency list
- `analysis.ipynb` — exploratory data analysis notebook

## Dependencies

The application is built for Python 3.12+ and depends on:

- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `plotly`
- `scikit-learn`
- `streamlit`

These packages are listed in `pyproject.toml`.

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install the required packages:

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib seaborn plotly scikit-learn streamlit
```

3. Ensure the `vgsales.csv` file is present in the project root.

## Run the Dashboard

Launch the Streamlit app with:

```bash
streamlit run app.py
```

Open the provided local URL in your browser to interact with the dashboard.

## Usage Notes

- Use the sidebar filters to adjust the year range, genre, and platform.
- Train the Random Forest model from the **Predictive Modeling** tab before making predictions.
- Download filtered results from the **Raw Data** tab as a CSV file.

## Recommended Workflow

1. Review trends in the **Market Overview** tab.
2. Compare categories in the **Deep Dive** tab.
3. Train and test the prediction model in the **Prediction Lab** tab.
4. Export data from the **Raw Data** tab for reports or presentation.

## Notes

- The app includes data cleaning steps such as missing year imputation and publisher fallback values.
- `main.py` is currently a simple placeholder and is not required to run the Streamlit app.

## License

This repository does not include a license file. Add a license if you plan to share or publish the project.
