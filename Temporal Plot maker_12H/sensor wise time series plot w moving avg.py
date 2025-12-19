import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
# Moving-average window size per sensor (in number of samples)
# Set 1 for no smoothing
# ============================================================

MA_WINDOW = {
    "pH": 20,
    "turbidity": 20,
    "temperature": 20,
    "conductivity": 20
}

# Example:
# MA_WINDOW = {"pH": 5, "turbidity": 20, "temperature": 10, "conductivity": 3}


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
# Helper – Moving Average Smoothing
# ============================================================

def smooth_series(series, window):
    """Return a moving-averaged version of the series."""
    if window <= 1:
        return series  # no smoothing
    return series.rolling(window=window, center=True, min_periods=1).mean()


# ============================================================
# 1. Read All CSVs
# ============================================================

dfs_raw = []
for loc_name, path in files:
    df = pd.read_csv(path)
    dfs_raw.append((loc_name, df))

# Detect columns from first dataset
time_col, ph_col, turb_col, temp_col, cond_col = detect_columns(dfs_raw[0][1])


# ============================================================
# 2. Convert timestamps, clean numeric, NO RESAMPLING
# ============================================================

dfs_aligned = []
min_start_time = None

for loc_name, df in dfs_raw:
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()

    for col in [ph_col, turb_col, temp_col, cond_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    dfs_aligned.append((loc_name, df))

    if min_start_time is None or df.index[0] < min_start_time:
        min_start_time = df.index[0]


# ============================================================
# 3. Compute hours-from-start for each dataset (no resampling)
# ============================================================

hours_list = []
for loc_name, df in dfs_aligned:
    hours = (df.index - min_start_time) / pd.Timedelta(hours=1)
    hours_list.append((loc_name, hours))


# ============================================================
# 4. Plot — 4 subplots (one per sensor), true value scales
# ============================================================

sensor_keys = ["pH", "turbidity", "temperature", "conductivity"]
sensor_cols = [ph_col, turb_col, temp_col, cond_col]

sensor_axis_colors = {
    "pH": COLORS["pH"],
    "turbidity": COLORS["turbidity"],
    "temperature": COLORS["temperature"],
    "conductivity": COLORS["conductivity"]
}

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(14, 10))
fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.10, hspace=0.32)

legend_lines = []
legend_labels = []

for ax, sensor_key, sensor_col in zip(axes, sensor_keys, sensor_cols):

    all_values = []

    for loc_idx, (loc_name, df) in enumerate(dfs_aligned):

        # raw series
        raw_series = df[sensor_col].astype(float)

        # smoothed series
        series = smooth_series(raw_series, MA_WINDOW[sensor_key])

        hours_series = hours_list[loc_idx][1]

        all_values.append(series)

        line, = ax.plot(
            hours_series, series,
            color=cc[loc_idx % len(cc)],  # location color
            label=loc_name,
            linewidth=1.3
        )

        if sensor_key == "pH":
            legend_lines.append(line)
            legend_labels.append(loc_name)

    # --- True Y-scale ---
    combined = pd.concat(all_values)
    vmin, vmax = combined.min(), combined.max()
    margin = 0.05 * (vmax - vmin if vmax > vmin else 1)
    ax.set_ylim(vmin - margin, vmax + margin)

    ax.set_ylabel(sensor_key)               # no color
    ax.tick_params(axis="y")                # default color (black)


    ax.grid(True, linestyle="--", alpha=0.4)

axes[-1].set_xlabel("Time (hours from start)")

# ============================================================
# X-axis formatting: ticks every 1 hour
# ============================================================

# Find global max hour across all locations
global_max_hour = 0
for loc_name, hours in hours_list:
    local_max = hours.max()
    if local_max > global_max_hour:
        global_max_hour = local_max

# Set ticks every 1 hour on the bottom axis
max_hour_int = int(np.ceil(global_max_hour))
axes[-1].set_xticks(np.arange(0, max_hour_int + 1, 1))
axes[-1].set_xlabel("Time (hours from start)")


# --- Combined Legend ---
fig.legend(
    handles=legend_lines,
    labels=legend_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    ncol=len(dfs_aligned),
    fontsize=10
)

plt.show()
