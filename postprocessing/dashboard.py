#%% 
import xarray as xr
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import dash
from dash import dcc, html, Input, Output

# === 📂 File paths ===
file_combined = r"C:\Users\ddo001\Documents\LoFloMeuse\Model_results\all_20250317_combined.nc"
file_discharge = r"C:\Users\ddo001\Documents\LoFloMeuse\Model_results\other\discharge_hourlyobs_smoothed.nc"

# === 📦 Load data ===
ds_combined = xr.open_dataset(file_combined)
ds_discharge = xr.open_dataset(file_discharge)
time_combined = pd.to_datetime(ds_combined.time.values)
time_discharge = pd.to_datetime(ds_discharge.time.values)

# === Dropdown options ===
all_indices = list(ds_combined.index.values.astype(str))
all_runs = list(ds_combined.runs.values.astype(str))
all_vars = list(ds_combined.data_vars.keys())
month_options = [t.strftime('%Y-%m') for t in pd.date_range(time_combined[0], time_combined[-1], freq='MS')]

# === 🌐 Dash App ===
app = dash.Dash(__name__)
server = app.server  # for deployment (optional)

app.layout = html.Div([
    html.H2("Hydro-Meteo Interactive Dashboard"),

    html.Div([
        html.Div([
            html.Label("Index:"),
            dcc.Dropdown(all_indices, all_indices[0], id="index-dropdown")
        ], style={"width": "20%", "display": "inline-block"}),

        html.Div([
            html.Label("Runs:"),
            dcc.Dropdown(all_runs, ["ref"], multi=True, id="runs-select")
        ], style={"width": "30%", "display": "inline-block"}),

        html.Div([
            html.Label("Variables:"),
            dcc.Dropdown(all_vars, ["Q"], multi=True, id="vars-select")
        ], style={"width": "30%", "display": "inline-block"}),
    ]),

    html.Div([
        html.Label("Time Range:"),
        dcc.RangeSlider(
            min=0, max=len(month_options) - 1,
            value=[0, len(month_options) - 1],
            marks={i: month for i, month in enumerate(month_options) if i % 12 == 0},
            tooltip={"placement": "bottom", "always_visible": False},
            id="time-slider"
        )
    ], style={"margin": "20px"}),

    html.Div([
        dcc.Checklist(
            options=[
                {"label": "Include Q_obs", "value": "obs"},
                {"label": "Include Q_hbv", "value": "hbv"},
            ],
            value=["obs", "hbv"],
            id="q-checklist",
            inline=True
        ),
        html.Button("Generate Time Series", id="btn-timeseries", n_clicks=0),
        html.Button("Generate Heatmap", id="btn-heatmap", n_clicks=0)
    ], style={"margin": "10px"}),

    dcc.Graph(id="main-plot")
])

# === 📊 Callback Logic ===
@app.callback(
    Output("main-plot", "figure"),
    Input("index-dropdown", "value"),
    Input("runs-select", "value"),
    Input("vars-select", "value"),
    Input("time-slider", "value"),
    Input("q-checklist", "value"),
    Input("btn-timeseries", "n_clicks"),
    Input("btn-heatmap", "n_clicks"),
)
def update_plot(index, runs, variables, time_range, q_opts, n_timeseries, n_heatmap):
    ctx = dash.callback_context
    t_start = pd.to_datetime(month_options[time_range[0]] + "-01")
    t_end = pd.to_datetime(month_options[time_range[1]] + "-01") + pd.offsets.MonthEnd(1)
    i_station = int((ds_combined.index == index).argmax().item())

    # Determine which button was pressed
    if ctx.triggered and ctx.triggered[0]['prop_id'].startswith("btn-heatmap"):
        return generate_heatmap(index, runs, variables, i_station, t_start, t_end)
    else:
        return generate_timeseries(index, runs, variables, i_station, t_start, t_end, q_opts)

# === 📈 Time Series Plot ===
def generate_timeseries(index, runs, variables, i_station, t0, t1, q_opts):
    fig = go.Figure()
    color_map = {
        'Pr': 'steelblue', 'Ea': 'orange', 'Q': 'darkred', 'Qgwr': 'blue',
        'Qinf': 'darkgreen', 'Qof': 'dodgerblue', 'Qssf': 'purple', 'Et': 'darkorange',
        'Sr': 'brown', 'St': 'teal'
    }

    if any(k in q_opts for k in ['obs', 'hbv']):
        i_q = int((ds_discharge.wflow_id == int(index)).argmax().item())
        df_q = pd.DataFrame({
            'Q_hbv': ds_discharge.Q[:, i_q, 0].values,
            'Q_obs': ds_discharge.Q[:, i_q, 1].values
        }, index=time_discharge).resample("D").mean().loc[t0:t1]

        if "obs" in q_opts:
            fig.add_trace(go.Scatter(x=df_q.index, y=df_q["Q_obs"], mode="lines", name="Q_obs", line=dict(color="black")))
        if "hbv" in q_opts:
            fig.add_trace(go.Scatter(x=df_q.index, y=df_q["Q_hbv"], mode="lines", name="Q_hbv", line=dict(color="green")))

    for var in variables:
        for run in runs:
            i_run = int((ds_combined.runs == run).argmax().item())
            data = ds_combined[var][:, i_station, i_run].to_series().loc[t0:t1]
            fig.add_trace(go.Scatter(
                x=data.index, y=data.values, mode='lines',
                name=f"{var} ({run})", line=dict(color=color_map.get(var, None), dash="dot" if var != "Pr" else "solid")
            ))

    fig.update_layout(title=f"Time Series | wflow_id={index}", xaxis_title="Date", yaxis_title="Value", hovermode="x unified")
    return fig

# === 🔥 Heatmap Plot ===
def generate_heatmap(index, runs, variables, i_station, t0, t1):
    means = {}
    for run in runs:
        i_run = int((ds_combined.runs == run).argmax().item())
        run_means = []
        for var in variables:
            data = ds_combined[var][:, i_station, i_run].to_series().loc[t0:t1]
            run_means.append(data.mean())
        means[run] = run_means

    df_means = pd.DataFrame(means, index=variables).T

    if "ref" in df_means.index:
        df_pct_change = (df_means - df_means.loc["ref"]) / df_means.loc["ref"] * 100
    else:
        df_pct_change = df_means.copy()

    fig = go.Figure(data=go.Heatmap(
        z=df_pct_change.values,
        x=variables,
        y=runs,
        colorscale='RdBu',
        zmid=0,
        colorbar=dict(title="% Change from ref"),
        hovertemplate="Run: %{y}<br>Var: %{x}<br>Change: %{z:.2f}%<extra></extra>"
    ))
    fig.update_layout(title=f"% Change from ref | wflow_id={index}", xaxis_title="Variable", yaxis_title="Run")
    return fig

# === 🏁 Run Server ===
if __name__ == "__main__":
    app.run_server(debug=True)

#%%