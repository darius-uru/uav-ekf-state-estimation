"""
run_uav_EKF.py

Driver script for UAV state estimation using an EKF library.

This script:
- Loads a CSV trajectory file
- Automatically detects available states:
    * Position only      -> EKF state [x, y, vx, vy]
    * Position + heading -> EKF state [x, y, vx, vy, psi, omega]
- Runs prediction/update
- Plots results
- Computes RMSE if truth columns exist

Author: Darius (ENAE380)
"""


# Setup / imports / path

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make sure local packages are visible
sys.path.append(".")

import EKF   #  EKF.py file with all of the helper functions, UAVstate class and methods



#choose CSV file here

CSV_FILE = "EKF_expo_data/expo1.csv"
# Examples:
# "EKF_expo_data/expo1.csv"
# "simulation_log/trajectory_log.csv"



# Load CSV

df = pd.read_csv(CSV_FILE)
print("Loaded:", CSV_FILE)
print("Columns:", list(df.columns))

# Required
t = df["t"].to_numpy(dtype=float)
x_meas = df["x_meas"].to_numpy(dtype=float)
y_meas = df["y_meas"].to_numpy(dtype=float)

# Optional heading states
HAS_HEADING = ("psi_meas" in df.columns) and ("omega_meas" in df.columns)

if HAS_HEADING:
    psi_meas = df["psi_meas"].to_numpy(dtype=float)
    omega_meas = df["omega_meas"].to_numpy(dtype=float)
    print("Detected heading data: using 6-state EKF")
else:
    print("No heading data detected: using 4-state EKF")

# Optional truth
HAS_TRUTH = ("x_true" in df.columns) and ("y_true" in df.columns)
if HAS_TRUTH:
    x_true = df["x_true"].to_numpy(dtype=float)
    y_true = df["y_true"].to_numpy(dtype=float)



# Measurement noise R

if HAS_TRUTH:
    R_pos = EKF.estimate_R_from_truth(x_true, y_true, x_meas, y_meas)
    print("R (position) estimated from truth")
else:
    sigma_xy = 10.0  # meters (tuneable)
    R_pos = np.diag([sigma_xy**2, sigma_xy**2])
    print("R (position) set manually")

# Heading noise (only if used)
if HAS_HEADING:
    sigma_psi = np.deg2rad(3.0)
    sigma_omega = np.deg2rad(5.0)
    R = np.diag([
        R_pos[0,0],
        R_pos[1,1],
        sigma_psi**2,
        sigma_omega**2
    ])
else:
    R = R_pos

print("R =\n", R)



# Initial state + covariance

if HAS_HEADING:
    x0 = [x_meas[0], y_meas[0], 0.0, 0.0, psi_meas[0], omega_meas[0]]
    P0 = np.diag([100, 100, 25, 25, 0.5, 1.0])
else:
    x0 = [x_meas[0], y_meas[0], 0.0, 0.0]
    P0 = np.diag([100, 100, 25, 25])

q_accel = 1.0   # motion smoothness knob
q_yaw   = 0.5   # heading maneuverability (only used if heading exists)



# Initialize EKF

if HAS_HEADING:
    ekf = EKF.UAV6DEKF(
        x0=x0,
        P0=P0,
        R=R,
        q_accel=q_accel,
        q_yaw=q_yaw
    )
else:
    ekf = EKF.UAV2DEKF(
        x0=x0,
        P0=P0,
        R=R,
        q_accel=q_accel
    )

print("EKF initialized")
print("Initial state:", x0)



# Run EKF

for k in range(1, len(t)):
    dt = t[k] - t[k-1]
    if dt <= 0:
        dt = 1e-6

    ekf.predict(dt)

    if HAS_HEADING:
        ekf.update([x_meas[k], y_meas[k], psi_meas[k], omega_meas[k]])
    else:
        ekf.update([x_meas[k], y_meas[k]])

X = ekf.get_posterior_states()
print("EKF complete. State shape:", X.shape)



# Plot position

plt.figure(figsize=(10,6))
plt.scatter(x_meas, y_meas, s=10, alpha=0.6, label="Measured")
plt.plot(X[:,0], X[:,1], linewidth=2, label="EKF estimate")
plt.xlabel("x")
plt.ylabel("y")
plt.title("UAV EKF Position Track")
plt.grid(True)
plt.legend()
plt.show()



# Plot heading (if present)

if HAS_HEADING:
    plt.figure(figsize=(10,4))
    plt.plot(t[:len(X)], X[:,4], label="Estimated ψ")
    plt.plot(t, psi_meas, "--", alpha=0.6, label="Measured ψ")
    plt.xlabel("time (s)")
    plt.ylabel("yaw (rad)")
    plt.title("Yaw Estimate")
    plt.grid(True)
    plt.legend()
    plt.show()



# Truth comparison + RMSE

if HAS_TRUTH:
    plt.figure(figsize=(10,6))
    plt.plot(x_true, y_true, "--", label="Truth")
    plt.plot(X[:,0], X[:,1], label="EKF estimate")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Truth vs EKF")
    plt.grid(True)
    plt.legend()
    plt.show()

    ex = X[:,0] - x_true[:len(X)]
    ey = X[:,1] - y_true[:len(X)]
    rmse = np.sqrt(np.mean(ex**2 + ey**2))
    print("Position RMSE:", rmse)
else:
    print("No truth columns available.")