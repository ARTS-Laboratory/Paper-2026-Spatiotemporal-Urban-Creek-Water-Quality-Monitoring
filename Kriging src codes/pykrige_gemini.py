import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging

# Set default fonts and plot colors

plt.rcParams.update({'text.usetex': True})
plt.rcParams.update({'image.cmap': 'viridis'})
plt.rcParams.update({'font.serif':['Times New Roman', 'Times', 'DejaVu Serif',
  'Bitstream Vera Serif', 'Computer Modern Roman', 'New Century Schoolbook',
  'Century Schoolbook L', 'Utopia', 'ITC Bookman', 'Bookman', 
  'Nimbus Roman No9 L', 'Palatino', 'Charter', 'serif']})
plt.rcParams.update({'font.family':'serif'})
plt.rcParams.update({'font.size': 10})
plt.rcParams.update({'mathtext.rm': 'serif'})
cc = plt.rcParams['axes.prop_cycle'].by_key()['color']

# =========================================================================
# 1. Synthetic Data Generation (Replace with your actual 4 data points)
# =========================================================================

# Your 4 known distances along the creek (x-coordinates)
distances = np.array([0.0, 10.0, 15, 27.0])  # e.g., in meters

# Your 4 measured pH values (z-coordinates)
# (This synthetic data is chosen to show a clear trend)
ph_values = np.array([6.8, 7.2, 7.7, 7.0])

# Define the grid of points for prediction (where we want to estimate pH)
# Create 100 equally spaced points between 0 and 50 meters
prediction_distances = np.linspace(distances.min(), distances.max(), 100)

# =========================================================================
# 2. Perform 1D Ordinary Kriging
# =========================================================================

# Initialize the Ordinary Kriging object
# Note: For 1D kriging, we must set the y-coordinates to a constant (e.g., 0.0)
# 'exponential': The variogram model we will fit (Spherical, Gaussian, etc., are other options)
# 'variogram_parameters': Set to None to let pykrige automatically estimate the parameters (Sill, Range, Nugget)
ok1d = OrdinaryKriging(
    distances,              # x-coordinates
    np.zeros(len(distances)), # y-coordinates (constant for 1D)
    ph_values,              # z-values (pH)
    variogram_model='gaussian',
    verbose=False,
    enable_plotting=False
)

# Execute the kriging process to get the predicted pH and the kriging variance
# The 'z' coordinates in the prediction step are also set to a constant (0.0)
ph_kriged, ph_kriged_std = ok1d.execute(
    'grid',
    prediction_distances,
    np.array([0.0]) # Must match the constant y-coordinate used earlier
)

# Extract the 1D results from the 2D array output (since pykrige is primarily 2D/3D)
ph_kriged = ph_kriged.data.flatten()
ph_kriged_std = ph_kriged_std.data.flatten()

# Calculate the Standard Error (Standard Deviation)
ph_kriged_se = np.sqrt(ph_kriged_std)

# =========================================================================
# 3. Visualization
# =========================================================================

# Define the confidence interval (e.g., 95% CI is roughly +/- 2 standard deviations)
confidence_interval = 2 * ph_kriged_se

plt.figure(figsize=(6.5, 3))



# Plot the Kriging Prediction (The estimated pH profile)
plt.plot(
    prediction_distances,
    ph_kriged,
    label='Kriging estimate',
   
)

# Plot the 95% confidence interval as a shaded area
plt.fill_between(
    prediction_distances,
    ph_kriged - confidence_interval,
    ph_kriged + confidence_interval,
    #color='skyblue',
    alpha=0.4,
    label='95% Confidence Interval'
)

# Plot the Known Data Points
plt.scatter(
    distances,
    ph_values,
    label='Measured Data point',

)

# Final Plot Touches

plt.xlabel('Distance Along Creek (meters)')
plt.ylabel('pH Value')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(y=7.0, linestyle=':', alpha=0.7, label='Neutral pH Reference')
plt.tight_layout(pad=0)
plt.show()