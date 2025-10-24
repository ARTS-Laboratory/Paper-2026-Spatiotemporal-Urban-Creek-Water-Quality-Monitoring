import numpy as np
import matplotlib.pyplot as plt
from gstools import Gaussian, krige
import pandas as pd

# ---- input (your data) ----
cond_pos = np.array([0.0, 10.0, 15, 27.0, 35])   # distances along creek (1D)
cond_val = np.array([6.8, 7.3, 8, 7.0, 7.3])         # pH readings

# prediction grid (1D)
gridx = np.linspace(0.0, 35.0, 100)

# ---- model & kriging object ----
model = Gaussian(dim=1, var=0.2, len_scale=2)
# Note: here we use Simple kriging as in your snippet
krig = krige.Simple(model, mean=1, cond_pos=cond_pos, cond_val=cond_val)

# ---- run kriging: capture returned (estimate, variance) ----
# Many gstools kriging objects return (field, var) when called
field, var = krig(gridx)

# numerical safety: ensure non-negative variance
var = np.clip(var, 0.0, None)
std = np.sqrt(var)

# 95% confidence interval (approx): mean ± 1.96 * std
ci_lower = field - 1.96 * std
ci_upper = field + 1.96 * std

# ---- plotting ----
fig, ax = plt.subplots(figsize=(9,4.5))
ax.plot(gridx, field, label="Kriging estimate", lw=1.8)
ax.fill_between(gridx, ci_lower, ci_upper, color="lightgrey", alpha=0.6,
                label="95% CI (±1.96·std)")
ax.scatter(cond_pos, cond_val, color="k", zorder=10, label="Conditions")
ax.set_xlabel("Distance along creek")
ax.set_ylabel("Value")
ax.set_title("1D Simple Kriging with 95% CI (gstools)")
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.show()

# ---- optional: save results to CSV ----
out = pd.DataFrame({
    "x": gridx,
    "estimate": field,
    "std": std,
    "ci_lower": ci_lower,
    "ci_upper": ci_upper
})
out.to_csv("gstools_kriging_with_ci.csv", index=False)
print("Saved results to 'gstools_kriging_with_ci.csv'")
