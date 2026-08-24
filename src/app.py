import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import EKF

st.title("UAV EKF Tester")

csv_file = st.text_input("CSV path", "data/trajectory_log.csv") # chgange to CSV that you want to implenment 

# Examples:
# "EKF_expo_data/expo1.csv"
# "simulation_log/trajectory_log.csv"

sigma_xy = st.slider("sigma_xy", 0.1, 50.0, 10.0, 0.1)
q_accel  = st.slider("q_accel", 0.001, 20.0, 1.0, 0.001)

if st.button("Run EKF"):
    df = pd.read_csv(csv_file)
    t = df["t"].to_numpy(float)
    x_meas = df["x_meas"].to_numpy(float)
    y_meas = df["y_meas"].to_numpy(float)

    R = np.diag([sigma_xy**2, sigma_xy**2])
    x0 = [x_meas[0], y_meas[0], 0.0, 0.0]
    P0 = np.diag([100,100,25,25])

    ekf = EKF.UAV2DEKF(x0=x0, P0=P0, R=R, q_accel=q_accel)

    for k in range(1, len(t)):
        dt = max(t[k]-t[k-1], 1e-6)
        ekf.predict(dt)
        ekf.update([x_meas[k], y_meas[k]])

    X = ekf.get_posterior_states()

    fig, ax = plt.subplots()
    ax.scatter(x_meas, y_meas, s=8, label="Measured")
    ax.plot(X[:,0], X[:,1], label="EKF")
    ax.grid(True); ax.legend()
    st.pyplot(fig)


# run with 
    # streamlit run app.py
# in terminal
