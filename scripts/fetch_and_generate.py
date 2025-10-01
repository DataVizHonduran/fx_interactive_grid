import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pandas_datareader import data
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from arch.unitroot import VarianceRatio
import warnings
from statsmodels.tools.sm_exceptions import InterpolationWarning
import json
import random

warnings.filterwarnings("ignore", category=InterpolationWarning)

def get_fred_fx(years=10, update=False, csv_file='data/fx_data.csv'):
    fx_labels = [
        "USDNOK", "USDSEK", "USDMXN", "USDBRL", "USDZAR",
        "USDINR", "USDKRW", "USDTHB", "USDSGD", "USDCNH",
        "USDJPY", "USDEUR", "USDGBP", "USDCAD", "USDCHF",
        "USDAUD", "USDNZD",
    ]
    
    if not update:
        print(f"Loading FX data from CSV: {csv_file}")
        df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
        print(f"Successfully loaded FX data from CSV: {df.shape}")
        return df
    
    datalist = [
        "DEXNOUS", "DEXSDUS", "DEXMXUS", "DEXBZUS", "DEXSFUS",
        "DEXINUS", "DEXKOUS", "DEXTHUS", "DEXSIUS", "DEXCHUS",
        "DEXJPUS", "DEXUSEU", "DEXUSUK", "DEXCAUS", "DEXSZUS",
        "DEXUSAL", "DEXUSNZ",
    ]
    
    end_date = datetime.date.today()
    print(f"Fetching FX data from FRED for date range ending: {end_date}")
    start_date = end_date - datetime.timedelta(days=365*years)
    df = data.DataReader(datalist, 'fred', start_date, end_date)
    df.columns = [
        "USDNOK", "USDSEK", "USDMXN", "USDBRL", "USDZAR",
        "USDINR", "USDKRW", "USDTHB", "USDSGD", "USDCNH",
        "USDJPY", "EURUSD", "GBPUSD", "USDCAD", "USDCHF",
        "AUDUSD", "NZDUSD",
    ]
    df = df.apply(pd.to_numeric, errors='coerce')
    df.index = pd.to_datetime(df.index)
    
    for old, new in [('EURUSD', 'USDEUR'), ('GBPUSD', 'USDGBP'), 
                     ('AUDUSD', 'USDAUD'), ('NZDUSD', 'USDNZD')]:
        if old in df.columns:
            df[new] = 1 / df[old]
            df = df.drop(old, axis=1)
    
    df = df.bfill()
    df.to_csv(csv_file)
    print(f"Saved FX data to CSV: {csv_file}")
    print(f"Successfully loaded FX data: {df.shape}")
    return df

def safe_adf(series):
    try:
        return adfuller(series, maxlag=10, autolag='AIC')[1]
    except Exception:
        return np.nan

def safe_kpss(series):
    try:
        return kpss(series, regression='c', nlags="auto")[1]
    except Exception:
        return np.nan

def safe_variance_ratio(series):
    try:
        return VarianceRatio(series).pvalue
    except Exception:
        return np.nan

def classify_series(series):
    series = series.dropna()
    if len(series) < 50:
        return 'insufficient_data'
    
    p_adf = safe_adf(series)
    p_kpss = safe_kpss(series)
    if pd.isna(p_adf) or pd.isna(p_kpss):
        return 'test_fail'
    
    if (p_adf < 0.05) and (p_kpss > 0.05):
        return 'mean_reverting'
    
    X = sm.add_constant(np.arange(len(series)))
    model = sm.OLS(series.values, X).fit()
    p_slope = model.pvalues[1] if len(model.pvalues) > 1 else np.nan
    p_vr = safe_variance_ratio(series)
    
    if p_slope < 0.05:
        return 'trending'
    elif (not pd.isna(p_vr)) and (p_vr > 0.05):
        return 'random_walk'
    else:
        return 'random_walk'

def find_cell_coords(pair, x_labels, y_labels):
    try:
        x0 = x_labels.get_loc(pair[1]) - 0.5
        x1 = x_labels.get_loc(pair[1]) + 0.5
        y0 = y_labels.get_loc(pair[0]) - 0.5
        y1 = y_labels.get_loc(pair[0]) + 0.5
        return x0, y0, x1, y1
    except KeyError:
        return None

# Main execution
print("Starting FX data update and dashboard generation...")

# Fetch new data
lookback = 10
df_prices = get_fred_fx(lookback, update=True)

window = 100

columns = df_prices.columns.tolist()
ratios = {}
for i, x in enumerate(columns):
    for y in columns[i+1:]:
        ratio_series = df_prices[x] / df_prices[y]
        last_window = ratio_series[-window:]
        mean_ = last_window.mean()
        std_ = last_window.std()
        last_value = last_window.iloc[-1]
        z_score = (last_value - mean_) / std_ if std_ != 0 else 0
        ratios[(x, y)] = {'ratio_series': ratio_series, 'z_score': z_score}

z_matrix = pd.DataFrame(np.nan, index=columns, columns=columns)
for (x,y), vals in ratios.items():
    z_matrix.loc[x,y] = vals['z_score']
    z_matrix.loc[y,x] = -vals['z_score']
np.fill_diagonal(z_matrix.values, 0)

# Classify pairs
results = {
    'mean_reverting': [],
    'trending': [],
    'random_walk': [],
    'insufficient_data': [],
    'test_fail': []
}

for pair, data in ratios.items():
    series = data['ratio_series']
    classification = classify_series(series)
    results[classification].append(pair)

# Find best pairs per currency
adf_results = {}
for pair, data in ratios.items():
    series = data['ratio_series'].dropna()
    try:
        pval = adfuller(series)[1]
    except:
        pval = 1
    adf_results[pair] = pval

currency_best_pairs = {}
for currency in z_matrix.index:
    relevant_pairs = [pair for pair in adf_results.keys() if currency in pair]
    if not relevant_pairs:
        continue
    best_pair = min(relevant_pairs, key=lambda p: adf_results[p])
    best_pval = adf_results[best_pair]
    currency_best_pairs[currency] = (best_pair, best_pval)

trend_pvalues = {}
for pair, data in ratios.items():
    series = data['ratio_series'].dropna()
    if len(series) < 20:
        continue
    X = sm.add_constant(np.arange(len(series)))
    model = sm.OLS(series.values, X).fit()
    pval = model.pvalues[1] if len(model.pvalues) > 1 else 1
    trend_pvalues[pair] = pval

currency_best_trending = {}
for currency in z_matrix.index:
    relevant_pairs = [pair for pair in trend_pvalues.keys() if currency in pair]
    if not relevant_pairs:
        continue
    best_pair = min(relevant_pairs, key=lambda p: trend_pvalues[p])
    best_pval = trend_pvalues[best_pair]
    currency_best_trending[currency] = (best_pair, best_pval)

mean_reverting_pairs = [pair for pair, pval in currency_best_pairs.values()]
trending_pairs = [pair for pair, pval in currency_best_trending.values()]

# ========== NEW: CREATE STANDALONE CHART FOR DEBUGGING ==========
print("\n" + "="*50)
print("Creating standalone test chart...")

# Pick a random pair
random_pair = random.choice(list(ratios.keys()))
print(f"Selected random pair: {random_pair[0]}/{random_pair[1]}")

random_ratio = ratios[random_pair]['ratio_series']
print(f"Ratio series length: {len(random_ratio)}")
print(f"Date range: {random_ratio.index[0]} to {random_ratio.index[-1]}")
print(f"Z-score: {ratios[random_pair]['z_score']:.3f}")

# Create standalone figure
standalone_fig = go.Figure()
standalone_fig.add_trace(go.Scatter(
    x=random_ratio.index,
    y=random_ratio.values,
    mode='lines',
    name=f'{random_pair[0]}/{random_pair[1]} Ratio',
    line=dict(color='blue', width=2)
))

# Add mean line
recent_mean = random_ratio[-window:].mean()
standalone_fig.add_hline(y=recent_mean, line_dash="dash", 
                         line_color="red", 
                         annotation_text=f"Mean (last {window} days)")

standalone_fig.update_layout(
    title=f"Test Chart: {random_pair[0]}/{random_pair[1]} Price Ratio",
    xaxis_title="Date",
    yaxis_title="Ratio",
    height=600,
    width=1200,
    showlegend=True
)

# Save standalone chart
standalone_html = standalone_fig.to_html()
with open('test_standalone_chart.html', 'w') as f:
    f.write(standalone_html)
print("✓ Saved test_standalone_chart.html")
print("="*50 + "\n")

# ========== END STANDALONE CHART ==========

# Create the main interactive figure
fig = make_subplots(
    rows=1, cols=2,
    shared_yaxes=False,
    horizontal_spacing=0.1,
    subplot_titles=(f"Z-Score Heatmap of Price Ratios (last {window} days)", "Price Ratio Over Time")
)

show_low_zs = True
z_masked = z_matrix.copy() if show_low_zs else z_matrix.where(z_matrix.abs() >= 1)

hover_text = z_masked.round(3).astype(str)
hover_text = hover_text.mask(z_masked.isna(), '')

heatmap = go.Heatmap(
    z=z_masked.values,
    x=z_masked.columns,
    y=z_masked.index,
    colorscale='RdYlGn',
    zmid=0,
    colorbar=dict(title='Z-Score'),
    hoverongaps=False,
    text=hover_text.values,
    hoverinfo='text+x+y'
)

default_pair = list(ratios.keys())[0]
default_ratio = ratios[default_pair]['ratio_series']

line = go.Scatter(
    x=default_ratio.index, 
    y=default_ratio.values, 
    mode='lines', 
    name=f'{default_pair[0]}/{default_pair[1]} Ratio'
)

fig.add_trace(heatmap, row=1, col=1)
fig.add_trace(line, row=1, col=2)

shapes = []

for pair in mean_reverting_pairs:
    coords = find_cell_coords(pair, z_masked.columns, z_masked.index)
    if coords:
        x0, y0, x1, y1 = coords
        shapes.append(dict(
            type='rect',
            xref='x',
            yref='y',
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(color='black', width=4, dash='solid'),
            fillcolor='rgba(0,0,0,0)',
            layer='above'
        ))

for pair in trending_pairs:
    coords = find_cell_coords(pair, z_masked.columns, z_masked.index)
    if coords:
        x0, y0, x1, y1 = coords
        shapes.append(dict(
            type='rect',
            xref='x',
            yref='y',
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(color='black', width=2, dash='dot'),
            fillcolor='rgba(0,0,0,0)',
            layer='above'
        ))

fig.update_layout(
    height=600,
    width=1400,
    title_text=f"FX Price Ratio Explorer - {lookback} year lookback (solid square = mean reverting, dotted square = trending)",
    shapes=shapes
)

# Prepare data for JavaScript - FIX NaN ISSUES
print("Preparing JSON data...")
ratios_json = {}
for (x, y), data in ratios.items():
    ratio_series = data['ratio_series'].dropna()  # Remove NaN values
    
    # Convert to list and handle any remaining NaN
    values_list = ratio_series.tolist()
    values_list = [float(v) if not np.isnan(v) else None for v in values_list]
    
    ratios_json[f"{x}|{y}"] = {
        'dates': ratio_series.index.strftime('%Y-%m-%d').tolist(),
        'values': values_list,
        'z_score': float(data['z_score']) if not np.isnan(data['z_score']) else 0
    }

print(f"Prepared {len(ratios_json)} ratio pairs for JSON")

# Convert Plotly figure to JSON properly
fig_json = json.loads(fig.to_json())

# Generate HTML with interactivity
html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FX Price Ratio Explorer</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1500px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .last-updated {{
            text-align: center;
            color: #666;
            font-size: 14px;
            margin-bottom: 20px;
        }}
        #chart {{
            width: 100%;
            height: 700px;
        }}
        .legend {{
            margin-top: 20px;
            padding: 15px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
            font-size: 14px;
        }}
        .legend-box {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid black;
            vertical-align: middle;
            margin-right: 5px;
        }}
        .solid {{
            border-style: solid;
            border-width: 3px;
        }}
        .dotted {{
            border-style: dotted;
            border-width: 2px;
        }}
        .debug-info {{
            position: fixed;
            bottom: 10px;
            right: 10px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 10px;
            border-radius: 5px;
            font-size: 12px;
            max-width: 300px;
            display: none;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>FX Price Ratio Explorer</h1>
        <div class="last-updated">
            Last Updated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
        </div>
        <div class="legend">
            <div class="legend-item">
                <span class="legend-box solid"></span>
                <span>Mean Reverting Pairs</span>
            </div>
            <div class="legend-item">
                <span class="legend-box dotted"></span>
                <span>Trending Pairs</span>
            </div>
            <div class="legend-item">
                <span>Click on heatmap cells to view ratio time series</span>
            </div>
        </div>
        <div id="chart"></div>
        <div id="debug" class="debug-info"></div>
    </div>

    <script>
        const ratiosData = {json.dumps(ratios_json)};
        const window_size = {window};

        const layout = {json.dumps(fig_json['layout'])};
        const data = {json.dumps(fig_json['data'])};
        
        console.log('Loaded ratios data:', Object.keys(ratiosData).length, 'pairs');
        console.log('Sample keys:', Object.keys(ratiosData).slice(0, 5));

        Plotly.newPlot('chart', data, layout, {{responsive: true}});

        const chartDiv = document.getElementById('chart');
        const debugDiv = document.getElementById('debug');
        
        // Show debug info on double-click
        chartDiv.addEventListener('dblclick', () => {{
            debugDiv.style.display = debugDiv.style.display === 'none' ? 'block' : 'none';
        }});
        
        chartDiv.on('plotly_click', function(eventData) {{
            try {{
                if (eventData.points[0].data.type === 'heatmap') {{
                    const xLabel = eventData.points[0].x;
                    const yLabel = eventData.points[0].y;
                    
                    console.log('Clicked:', xLabel, 'x', yLabel);
                    
                    let key = xLabel + '|' + yLabel;
                    let reverse = false;
                    
                    if (!ratiosData[key]) {{
                        key = yLabel + '|' + xLabel;
                        reverse = true;
                        console.log('Using reversed key:', key);
                    }}
                    
                    if (ratiosData[key]) {{
                        const pairData = ratiosData[key];
                        let values = pairData.values.filter(v => v !== null);
                        
                        console.log('Found data:', values.length, 'points');
                        
                        // Calculate mean from recent values
                        const recentValues = values.slice(-window_size);
                        const meanRecent = recentValues.reduce((a, b) => a + b, 0) / recentValues.length;
                        
                        console.log('Recent mean:', meanRecent);
                        
                        // Auto-invert if mean < 1
                        let titleSuffix = '';
                        if (meanRecent < 1) {{
                            values = values.map(v => 1 / v);
                            titleSuffix = ' (inverted)';
                            console.log('Inverted ratio');
                        }}
                        
                        const update = {{
                            x: [pairData.dates],
                            y: [values],
                            name: [`${{xLabel}}/${{yLabel}} Ratio${{titleSuffix}}`]
                        }};
                        
                        Plotly.restyle('chart', update, [1]);
                        
                        layout.annotations[1].text = `Price Ratio ${{xLabel}}/${{yLabel}} Over Time${{titleSuffix}}`;
                        Plotly.relayout('chart', {{'annotations': layout.annotations}});
                        
                        // Update debug info
                        debugDiv.innerHTML = `
                            <strong>Last Click:</strong><br>
                            Pair: ${{xLabel}}/${{yLabel}}<br>
                            Key: ${{key}}<br>
                            Points: ${{values.length}}<br>
                            Mean: ${{meanRecent.toFixed(4)}}<br>
                            Z-score: ${{pairData.z_score.toFixed(3)}}<br>
                            Inverted: ${{titleSuffix ? 'Yes' : 'No'}}
                        `;
                        debugDiv.style.display = 'block';
                        
                        console.log('Chart updated successfully');
                    }} else {{
                        console.error('Key not found:', key);
                        debugDiv.innerHTML = `<strong>Error:</strong><br>Key not found: ${{key}}`;
                        debugDiv.style.display = 'block';
                    }}
                }}
            }} catch(error) {{
                console.error('Error in click handler:', error);
                debugDiv.innerHTML = `<strong>Error:</strong><br>${{error.message}}`;
                debugDiv.style.display = 'block';
            }}
        }});
        
        console.log('Chart initialized successfully');
    </script>
</body>
</html>
"""

# Save HTML file
with open('index.html', 'w') as f:
    f.write(html_template)

print("\n" + "="*50)
print("Dashboard generated successfully!")
print(f"Last update: {datetime.datetime.now()}")
print("Files created:")
print("  - index.html (interactive dashboard)")
print("  - test_standalone_chart.html (standalone test chart)")
print("\nDebugging tips:")
print("  - Open test_standalone_chart.html to verify basic Plotly functionality")
print("  - Open index.html and double-click anywhere to toggle debug info")
print("  - Check browser console (F12) for JavaScript errors")
print("="*50)

warnings.filterwarnings("default", category=InterpolationWarning)
