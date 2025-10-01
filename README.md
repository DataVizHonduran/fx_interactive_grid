# FX Price Ratio Explorer

Interactive dashboard analyzing foreign exchange price ratios and identifying mean-reverting and trending currency pairs.

## Features

- Daily automated data updates from FRED
- Interactive heatmap of Z-scores for currency pair ratios
- Statistical classification of mean-reverting and trending pairs
- Click-to-explore time series visualization

## Data Update Schedule

Data is automatically updated daily at 12:00 UTC via GitHub Actions.

## View Dashboard

Visit: `https://DataVizHonduran.github.io/fx_interactive_grid/`

## Local Development
```bash
pip install -r requirements.txt
python scripts/fetch_and_generate.py
