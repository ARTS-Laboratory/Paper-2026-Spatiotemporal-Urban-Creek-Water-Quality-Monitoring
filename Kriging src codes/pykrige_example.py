"""
1D Ordinary Kriging using PyKrige
- Replace `positions` and `values` with your 4 measured locations and pH readings.
- Produces kriging estimate along a line, plus 95% CI band, and saves results to CSV.
Requires: numpy, matplotlib, pandas, pykrige
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ---- USER INPUT: replace with your data ----
positions = np.array([0.0, 10.0, 15, 27.0, 35])   # distances along creek (1D)
values    = np.array([6.8, 7.3, 8, 7.0, 7.3])         # pH readings
# --------------------------------------------

# create y=0 for all points so we can use the 2D API as a 1D line
x = positions
y = np.zeros_like(x)

# prediction locations along the same line (dense grid for smooth curve)
x_pred = np.linspace(x.min(), x.max(), 500)
y_pred = np.zeros_like(x_pred)

# --- PyKrige ordinary kriging ---
from pykrige.ok import OrdinaryKriging

# Create the kriging object.
# variogram_model can be 'linear', 'power', 'gaussian', 'spherical', 'exponential' or a callable.
# If variogram_model_parameters is None (default), pykrige will attempt to fit parameters automatically.
OK = OrdinaryKriging(
    x, y, values,
    variogram_model='exponential',   # choose model; will be fitted to the data
    verbose=False,
    enable_plotting=False
)

# You can execute kriging for a list of points with `execute('points', xpoints, ypoints)`
z_pred, ss_pred = OK.execute('points', x_pred, y_pred)
# z_pred: predicted mean; ss_pred: kriging variance

# Ensure shapes are 1D arrays
z_pred = np.array(z_pred).flatten()
ss_pred = np.array(ss_pred).flatten()
ss_pred = np.clip(ss_pred, 0.0, None)  # numerical safety
std_pred = np.sqrt(ss_pred)

# 95% CI: mean ± 1.96 * std
ci_lower = z_pred - 1.9 * std_pred
ci_upper = z_pred + 1.9 * std_pred

# ---------------- PLOTTING ----------------
plt.figure(figsize=(10,5))
plt.plot(x_pred, z_pred, color='tab:blue', lw=1.8, label='Kriging estimate')
plt.fill_between(x_pred, ci_lower, ci_upper, color='lightgrey', alpha=0.6, label='95% CI')
plt.scatter(x, values, color='k', zorder=5, s=60, label='Observations')
plt.xlabel('Distance along creek')
plt.ylabel('pH')
plt.title('1D Ordinary Kriging (PyKrige) — estimate with 95% CI')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------- SAVE RESULTS ----------------
out_df = pd.DataFrame({
    'x': x_pred,
    'pH_pred': z_pred,
    'pH_std': std_pred,
    'ci_lower': ci_lower,
    'ci_upper': ci_upper
})
out_df.to_csv('pykrige_kriging_1d_results.csv', index=False)
print("Saved results to 'pykrige_kriging_1d_results.csv'")

