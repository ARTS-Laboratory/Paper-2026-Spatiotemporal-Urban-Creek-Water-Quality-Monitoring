import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Extract cycle colors
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Assign your variables dynamically
COLORS = {
    "pH": cc[0],
    "turbidity": cc[1],
    "temperature": cc[2],
    "conductivity (µS/cm)": cc[3]
}

# ============================================================
# CONFIG
# ============================================================

# Sampling interval (hours) for resampling
RESAMPLE_HOURS = 0.5   # change this to whatever you want (e.g. 1.0, 0.25, etc.)

# File names and location labels
files = [
    ("Location 1", "time_trimmed_data\ONE.CSV"),
    ("Location 2", "time_trimmed_data\TWO.CSV"),
    ("Location 3", "time_trimmed_data\THREE.CSV"),
    ("Location 4", "time_trimmed_data\FOUR.CSV"),
]


# ============================================================
# Helper: detect columns by name (case-insensitive, flexible)
# ============================================================

def detect_columns(df):
    cols = df.columns

    def find(substring_options):
        substring_options = [substring_options] if isinstance(substring_options, str) else substring_options
        for c in cols:
            cl = c.lower()
            for s in substring_options:
                if s in cl:
                    return c
        raise KeyError(f"Could not find a column containing any of {substring_options} in {list(cols)}")

    time_col = find(["time", "date"])        # date-time column
    ph_col = find("ph")                      # pH
    turb_col = find("turb")                  # turbidity
    temp_col = find("temp")                  # temperature
    cond_col = find(["cond", "conduct"])     # conductivity (µS/cm)

    return time_col, ph_col, turb_col, temp_col, cond_col

# Normalize function for plotting
def normalize_series(s):
    s = s.astype(float)
    smin = s.min()
    smax = s.max()
    if pd.isna(smin) or pd.isna(smax) or smin == smax:
        # Avoid division by zero; fall back to 0.5
        return pd.Series(np.full_like(s, 0.5, dtype=float), index=s.index), smin, smax
    norm = (s - smin) / (smax - smin)
    return norm, smin, smax

# Create tick positions and label values
def make_ticks(vmin, vmax, n_ticks=6):
    if pd.isna(vmin) or pd.isna(vmax):
        return [0.5], [""]   # fallback
    ticks = np.linspace(vmin, vmax, n_ticks)
    labels = [f"{t:.2f}" for t in ticks]
    positions = np.linspace(0, 1, n_ticks)   # normalized positions
    return positions, labels

# ============================================================
# 1. Read all CSVs
# ============================================================

dfs_raw = []
for loc_name, path in files:
    df = pd.read_csv(path)
    dfs_raw.append((loc_name, df))

# Detect column names from the first file (assumed same for all)
time_col, ph_col, turb_col, temp_col, cond_col = detect_columns(dfs_raw[0][1])

# ============================================================
# 2. Convert time to datetime, set index, sort, resample
# ============================================================

resample_minutes = int(RESAMPLE_HOURS * 60)
RESAMPLE_FREQ = f"{resample_minutes}T"  # e.g. "30T" for 0.5 hours

dfs_resampled = []
for loc_name, df in dfs_raw:
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()

    # Ensure numeric
    for col in [ph_col, turb_col, temp_col, cond_col]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Resample
    df_res = df.resample(RESAMPLE_FREQ).mean()
    dfs_resampled.append((loc_name, df_res))

# ============================================================
# 3. Align all locations to the same time range (intersection)
# ============================================================

common_index = dfs_resampled[0][1].index
for _, df in dfs_resampled[1:]:
    common_index = common_index.intersection(df.index)

dfs_aligned = []
for loc_name, df in dfs_resampled:
    df_aligned = df.loc[common_index].copy()
    dfs_aligned.append((loc_name, df_aligned))

if len(common_index) == 0:
    raise ValueError("No common timestamps across all files after resampling.")

# Convert index (datetime) to hours from zero
t0 = common_index[0]
hours = (common_index - t0) / pd.Timedelta(hours=1)
total_hours = hours.max()

# ============================================================
# 4. Plot: 4 subplots, each with 4 y-axes (2 left, 2 right),
#    all sharing the same grid/tick positions via normalization
# ============================================================

n_locations = len(dfs_aligned)
fig, axes = plt.subplots(n_locations, 1, sharex=True, figsize=(6.5, 6.5))
if n_locations == 1:
    axes = [axes]

#fig.subplots_adjust(left=0.3, right=0.7, hspace=0.35)

# To store legend items (only once)
legend_lines = []
legend_labels = []
collected_legend = False

for idx, (ax, (loc_name, df)) in enumerate(zip(axes, dfs_aligned)):
    # Extract variables
    ph = df[ph_col]
    turb = df[turb_col]
    temp = df[temp_col]
    cond = df[cond_col]

    # Normalize each series for plotting
    ph_norm, ph_min, ph_max = normalize_series(ph)
    turb_norm, turb_min, turb_max = normalize_series(turb)
    temp_norm, temp_min, temp_max = normalize_series(temp)
    cond_norm, cond_min, cond_max = normalize_series(cond)

    # -----------------------------
    # pH AXIS (LEFT)
    # -----------------------------
    ax.set_ylim(0, 1)
    ph_line, = ax.plot(hours, ph_norm, label="pH", color=COLORS["pH"])
    ax.set_ylabel("pH", color=COLORS["pH"])
    ax.tick_params(axis="y", colors=COLORS["pH"])

    ph_tick_pos, ph_tick_lbl = make_ticks(ph_min, ph_max)
    ax.set_yticks(ph_tick_pos)
    ax.set_yticklabels(ph_tick_lbl)

    ax.grid(True, which="both", axis="both")

    # -----------------------------
    # TURBIDITY AXIS (SECOND LEFT)
    # -----------------------------
    ax_turb = ax.twinx()
    ax_turb.spines["left"].set_position(("axes", -0.12))
    ax_turb.spines["left"].set_visible(True)
    ax_turb.spines["right"].set_visible(False)
    ax_turb.yaxis.set_label_position("left")
    ax_turb.yaxis.set_ticks_position("left")

    ax_turb.set_ylim(0, 1)
    turb_line, = ax_turb.plot(hours, turb_norm, label="Turbidity", color=COLORS["turbidity"])
    ax_turb.set_ylabel("Turbidity (NTU)", color=COLORS["turbidity"])
    ax_turb.tick_params(axis="y", colors=COLORS["turbidity"])

    turb_tick_pos, turb_tick_lbl = make_ticks(turb_min, turb_max)
    ax_turb.set_yticks(turb_tick_pos)
    ax_turb.set_yticklabels(turb_tick_lbl)

    # -----------------------------
    # TEMPERATURE AXIS (FIRST RIGHT)
    # -----------------------------
    ax_temp = ax.twinx()
    ax_temp.set_ylim(0, 1)
    temp_line, = ax_temp.plot(hours, temp_norm, label="Temperature", color=COLORS["temperature"])
    ax_temp.set_ylabel("temperature (°C)", color=COLORS["temperature"])
    ax_temp.tick_params(axis="y", colors=COLORS["temperature"])

    temp_tick_pos, temp_tick_lbl = make_ticks(temp_min, temp_max)
    ax_temp.set_yticks(temp_tick_pos)
    ax_temp.set_yticklabels(temp_tick_lbl)

    # -----------------------------
    # conductivity (µS/cm) AXIS (SECOND RIGHT)
    # -----------------------------
    ax_cond = ax.twinx()
    ax_cond.spines["right"].set_position(("axes", 1.12))
    ax_cond.spines["right"].set_visible(True)
    ax_cond.yaxis.set_label_position("right")
    ax_cond.yaxis.set_ticks_position("right")

    ax_cond.set_ylim(0, 1)
    cond_line, = ax_cond.plot(hours, cond_norm, label="conductivity (µS/cm)", color=COLORS["conductivity (µS/cm)"])
    ax_cond.set_ylabel("conductivity (µS/cm)", color=COLORS["conductivity (µS/cm)"])
    ax_cond.tick_params(axis="y", colors=COLORS["conductivity (µS/cm)"])

    cond_tick_pos, cond_tick_lbl = make_ticks(cond_min, cond_max)
    ax_cond.set_yticks(cond_tick_pos)
    ax_cond.set_yticklabels(cond_tick_lbl)

    # -----------------------------
    # TITLE + X LIMITS
    # -----------------------------
    ax.set_xlim(0, total_hours)
    ax.set_title(loc_name)

    # -----------------------------
    # ONE-TIME LEGEND COLLECTION
    # -----------------------------
    if not collected_legend:
        legend_lines = [ph_line, turb_line, temp_line, cond_line]
        legend_labels = [l.get_label() for l in legend_lines]
        collected_legend = True

# ============================================================
# FINAL LEGEND (ONLY ONCE, ABOVE FIRST PLOT)
# ============================================================
fig.legend(
    legend_lines,
    legend_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1),
    ncol=4,
    fontsize=10
)


# ============================================================
# 5. X-axis formatting: hours from 0 to end, 1-hour ticks
# ============================================================

max_hour_int = int(np.ceil(total_hours))
axes[-1].set_xlabel("Time (hours from start)")
axes[-1].set_xticks(np.arange(0, max_hour_int + 1, 1))

#plt.tight_layout(pad=0)

fig.subplots_adjust(
    left=0.175,
    right=0.825,
    top=0.90,
    bottom=0.08,
    hspace=0.30
)
plt.show()
