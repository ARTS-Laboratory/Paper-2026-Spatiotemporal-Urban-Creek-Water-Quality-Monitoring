import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

# ============================================================
# Matplotlib Style Configuration
# ============================================================

plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'image.cmap': 'viridis'})
plt.rcParams.update({'font.serif': [
    'Times New Roman', 'Times', 'DejaVu Serif', 'Bitstream Vera Serif',
    'Computer Modern Roman', 'New Century Schoolbook',
    'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman',
    'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif'
]})
plt.rcParams.update({'font.family': 'serif'})
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'mathtext.rm': 'serif'})
plt.rcParams.update({'mathtext.fontset': 'custom'})

# Extract cycle colors for LOCATIONS
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Sensor colors (axis label coloring)
COLORS = {
    "pH": cc[0],
    "turbidity": cc[1],
    "temperature": cc[2],
    "conductivity": cc[3]
}

# ============================================================
# Configuration
# ============================================================

files = [
    ("Location 1", "time_trimmed_data/ONE.CSV"),
    ("Location 2", "time_trimmed_data/TWO.CSV"),
    ("Location 3", "time_trimmed_data/THREE.CSV"),
    ("Location 4", "time_trimmed_data/FOUR.CSV"),
]

# ============================================================
# Smoothing Parameters
# ============================================================

MA_WINDOW = {
    "pH": 20,
    "turbidity": 20,
    "temperature": 20,
    "conductivity": 20
}


# ============================================================
# Helper – Detect Columns
# ============================================================

def detect_columns(df):
    cols = df.columns

    def find(sub):
        if isinstance(sub, str):
            sub = [sub]
        for c in cols:
            cl = c.lower()
            for s in sub:
                if s.lower() in cl:
                    return c
        raise KeyError(f"Could not find column containing {sub}")

    time_col = find(["time", "date"])
    ph_col   = find("ph")
    turb_col = find("turb")
    temp_col = find("temp")
    cond_col = find(["cond", "conduct"])

    return time_col, ph_col, turb_col, temp_col, cond_col


# ============================================================
# Helper – Moving Average
# ============================================================

def smooth_series(series, window):
    if window <= 1:
        return series
    return series.rolling(window=window, center=True, min_periods=1).mean()


# ============================================================
# 1. Read CSV Files
# ============================================================

dfs_raw = []
for loc_name, path in files:
    df = pd.read_csv(path)
    dfs_raw.append((loc_name, df))

# Detect column headers
time_col, ph_col, turb_col, temp_col, cond_col = detect_columns(dfs_raw[0][1])


# ============================================================
# 2. Convert timestamps, NO RESAMPLING, remove timezone
# ============================================================

dfs_aligned = []
min_start_time = None

for loc_name, df in dfs_raw:
    df[time_col] = pd.to_datetime(df[time_col]).dt.tz_localize(None)
    df = df.set_index(time_col).sort_index()

    for col in [ph_col, turb_col, temp_col, cond_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    dfs_aligned.append((loc_name, df))

    if min_start_time is None or df.index[0] < min_start_time:
        min_start_time = df.index[0]


# ============================================================
# 3. Compute HOURS FROM START (sensor data)
# ============================================================

hours_list = []
for loc_name, df in dfs_aligned:
    hours = (df.index - min_start_time) / pd.Timedelta(hours=1)
    hours_list.append((loc_name, hours))


# ============================================================
# 4. Download & Prepare USGS Stage Height Data
# ============================================================

site = "02169506"
parameter = "00065"        # stage height feet
start = "2025-12-04T22:15:00-05:00"
end   = "2025-12-05T10:15:00-05:00"

url = "https://waterservices.usgs.gov/nwis/iv/"
params = {
    "format": "json",
    "sites": site,
    "parameterCd": parameter,
    "startDT": start,
    "endDT": end,
    "siteStatus": "all"
}

response = requests.get(url, params=params)
response.raise_for_status()
data = response.json()

ts = data["value"]["timeSeries"][0]
values = ts["values"][0]["value"]

df_usgs = pd.DataFrame(values)
df_usgs["dateTime"] = pd.to_datetime(df_usgs["dateTime"]).dt.tz_localize(None)
df_usgs["value"] = pd.to_numeric(df_usgs["value"])
df_usgs.set_index("dateTime", inplace=True)

# Convert FEET → METERS
df_usgs["value_m"] = df_usgs["value"] * 0.3048

# USGS hours-from-start
usgs_hours = (df_usgs.index - min_start_time) / pd.Timedelta(hours=1)


# ============================================================
# 5. PLOT: 4 sensor subplots + 1 USGS subplot
# ============================================================

fig, axes = plt.subplots(5, 1, sharex=True, figsize=(14, 12))
fig.subplots_adjust(left=0.10, right=0.95, top=0.92, bottom=0.10, hspace=0.32)

sensor_keys = ["pH", "turbidity", "temperature", "conductivity"]
sensor_cols = [ph_col, turb_col, temp_col, cond_col]

legend_lines = []
legend_labels = []

# ---- Sensor Plots (Subplots 1 to 4) ----
for ax, sensor_key, sensor_col in zip(axes[:4], sensor_keys, sensor_cols):

    all_values = []

    for loc_idx, (loc_name, df) in enumerate(dfs_aligned):

        raw_series = df[sensor_col].astype(float)
        series = smooth_series(raw_series, MA_WINDOW[sensor_key])
        hours_series = hours_list[loc_idx][1]

        all_values.append(series)

        line, = ax.plot(
            hours_series,
            series,
            color=cc[loc_idx % len(cc)],
            label=loc_name,
            linewidth=1.3
        )

        if sensor_key == "pH":
            legend_lines.append(line)
            legend_labels.append(loc_name)

    combined = pd.concat(all_values)
    vmin, vmax = combined.min(), combined.max()
    margin = 0.05 * (vmax - vmin if vmax > vmin else 1)
    ax.set_ylim(vmin - margin, vmax + margin)
    ax.set_ylabel(sensor_key)
    ax.grid(True, linestyle="--", alpha=0.4)


# ---- USGS Stage Height (Subplot 5) ----
ax_usgs = axes[4]
ax_usgs.plot(
    usgs_hours,
    df_usgs["value_m"],
    color="black",
    linewidth=1.4,
    label="USGS Stage Height"
)

ax_usgs.set_ylabel("Stage (m)")
ax_usgs.set_xlabel("Time (hours from start)")
ax_usgs.grid(True, linestyle="--", alpha=0.4)


# ---- Combined Legend ----
fig.legend(
    handles=legend_lines,
    labels=legend_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    ncol=len(dfs_aligned),
    fontsize=10
)


# ============================================================
# 6. X-axis hourly ticks (shared)
# ============================================================

# Maximum hour across sensor and USGS data
global_max_hour = 0

for loc_name, h in hours_list:
    global_max_hour = max(global_max_hour, h.max())

global_max_hour = max(global_max_hour, usgs_hours.max())

max_hour_int = int(np.ceil(global_max_hour))

axes[-1].set_xticks(np.arange(0, max_hour_int + 1, 1))

plt.show()
