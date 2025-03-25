#%% 

import xarray as xr
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import dash
from dash import dcc, html, Input, Output
import os
import requests

# === 📂 SURFdrive File URLs ===
url_combined = "https://surfdrive.surf.nl/files/remote.php/webdav/LoFloMeuse_Deltares/Handover_190325/all_20250317_combined.nc"
url_discharge = "https://surfdrive.surf.nl/files/remote.php/webdav/LoFloMeuse_Deltares/Handover_190325/discharge_hourlyobs_smoothed.nc"

file_combined = "/tmp/all_20250317_combined.nc"
file_discharge = "/tmp/discharge_hourlyobs_smoothed.nc"

# === 🔐 Download with WebDAV credentials from environment ===
user = os.getenv("SURF_USER")
password = os.getenv("SURF_PASS")


def download_file(url, dest_path):
    if not os.path.exists(dest_path):  # Only download if not already present
        response = requests.get(url, auth=(user, password))
        if response.status_code == 200:
            with open(dest_path, 'wb') as f:
                f.write(response.content)
        else:
            raise Exception(f"Failed to download {url}: {response.status_code} - {response.text}")

download_file(url_combined, file_combined)
download_file(url_discharge, file_discharge)

# === 📦 Load data ===
ds_combined = xr.open_dataset(file_combined)
ds_discharge = xr.open_dataset(file_discharge)
time_combined = pd.to_datetime(ds_combined.time.values)
time_discharge = pd.to_datetime(ds_discharge.time.values)

all_indices = list(ds_combined.index.values.astype(str))
all_runs = list(ds_combined.runs.values.astype(str))
all_vars = list(ds_combined.data_vars.keys())
month_options = [t.strftime('%Y-%m') for t in pd.date_range(time_combined[0], time_combined[-1], freq='MS')]

# === 🌐 Dash App ===
app = dash.Dash(__name__, title="Hydro-Meteo Dashboard")
server = app.server

app.layout = html.Div([
    html.H1("Hydro-Meteo Interactive Dashboard", style={"textAlign": "center", "marginBottom": 30}),

    html.Div([
        html.Div([
            html.Label("Select Index (wflow_id):"),
            dcc.Dropdown(all_indices, all_indices[0], id="index-dropdown")
        ], className="three columns"),

        html.Div([
            html.Label("Select Runs:"),
            dcc.Dropdown(all_runs, ["ref"], multi=True, id="runs-select")
        ], className="four columns"),

        html.Div([
            html.Label("Select Variables:"),
            dcc.Dropdown(all_vars, ["Q"], multi=True, id="vars-select")
        ], className="five columns")
    ], className="row", style={"padding": "0 20px"}),

    html.Div([
        html.Label("Select Time Range:"),
        dcc.RangeSlider(
            min=0, max=len(month_options) - 1,
            value=[0, len(month_options) - 1],
            marks={i: month for i, month in enumerate(month_options) if i % 12 == 0},
            tooltip={"placement": "bottom"},
            id="time-slider"
        )
    ], style={"margin": "40px 20px"}),

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
        html.Button("Generate Time Series", id="btn-timeseries", n_clicks=0, style={"marginLeft": "20px"}),
        html.Button("Generate Heatmap", id="btn-heatmap", n_clicks=0, style={"marginLeft": "10px"})
    ], style={"padding": "0 20px", "marginBottom": 20}),

    dcc.Loading(
        dcc.Graph(id="main-plot"),
        type="default",
        style={"padding": "0 20px"}
    )
])

@app.callback(
    Output("main-plot", "figure"),
    Input("index-dropdown", "value"),
    Input("runs-select", "value"),
    Input("vars-select", "value"),
    Input("time-slider", "value"),
    Input("q-checklist", "value"),
    Input("btn-timeseries", "n_clicks"),
    Input("btn-heatmap", "n_clicks")
)
def update_plot(index, runs, variables, time_range, q_opts, n_timeseries, n_heatmap):
    ctx = dash.callback_context
    t_start = pd.to_datetime(month_options[time_range[0]] + "-01")
    t_end = pd.to_datetime(month_options[time_range[1]] + "-01") + pd.offsets.MonthEnd(1)
    i_station = int((ds_combined.index == index).argmax().item())

    if ctx.triggered and ctx.triggered[0]['prop_id'].startswith("btn-heatmap"):
        return generate_heatmap(runs, variables, i_station, t_start, t_end)
    else:
        return generate_timeseries(index, runs, variables, i_station, t_start, t_end, q_opts)

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

    fig.update_layout(
        title=f"Time Series | wflow_id={index}",
        xaxis_title="Date",
        yaxis_title="Value",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=40, b=40)
    )
    return fig

def generate_heatmap(runs, variables, i_station, t0, t1):
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

    fig.update_layout(
        title=f"% Change from ref | wflow_id={all_indices[i_station]}",
        xaxis_title="Variable",
        yaxis_title="Run",
        margin=dict(l=40, r=20, t=40, b=40)
    )
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
#%%