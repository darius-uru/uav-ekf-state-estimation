
# EKF_UAV.py

# Purpose:
#   A reusable EKF-style state estimator for a UAV moving in 2D.
#
#   This module is meant to be imported like a library:
#       from EKF_UAV import UAV2DEKF
#
# What it estimates:
#   State vector: x = [x, y, vx, vy]^T
#     - x, y   : position in the plane
#     - vx, vy : velocity components
#
# What it assumes you measure:
#   Measurement vector: z = [x_meas, y_meas]^T
#     - typical GPS-like position measurement
#
# Why EKF-style if the model is linear?
#   This is technically a standard Kalman Filter (KF) because:
#     - motion model is linear
#     - measurement model is linear
#
#   BUT we keep the EKF predict/update structure so you can later:
#     - add yaw (psi) / yaw rate
#     - add IMU accel/gyro measurements
#     - use nonlinear motion models
# ============================================================

import numpy as np


def _as_col(v):
    """
    Convert v into a column vector with shape (n, 1).

    WHY:
      EKF equations are easiest + least bug-prone if all vectors
      are columns. This prevents shape mistakes like (4,) vs (4,1).
    """
    v = np.array(v, dtype=float)
    if v.ndim == 1:
        return v.reshape((-1, 1))
    if v.ndim == 2 and v.shape[0] == 1:
        return v.T
    return v


def estimate_R_from_truth(x_true, y_true, x_meas, y_meas, floor=1e-6):
    """
    Estimate measurement covariance R from truth vs measurement.

    Inputs:
      x_true, y_true : "ground truth" positions
      x_meas, y_meas : measured positions
      floor          : small number to avoid zero variance

    Output:
      R = diag(var(x_meas-x_true), var(y_meas-y_true))

    WHY:
      If your CSV contains truth columns, you can justify R with data.
      This is a VERY defensible design decision in a report/video.
    """
    x_true = np.array(x_true, dtype=float)
    y_true = np.array(y_true, dtype=float)
    x_meas = np.array(x_meas, dtype=float)
    y_meas = np.array(y_meas, dtype=float)

    ex = x_meas - x_true
    ey = y_meas - y_true

    vx = float(np.var(ex)) + floor
    vy = float(np.var(ey)) + floor

    return np.diag([vx, vy])


class UAV2DEKF:
    """
    EKF-style state estimator for planar UAV motion.

    State:
      x = [x, y, vx, vy]^T # four vector of state imput

    Motion model (constant velocity):
      x_k  = x_{k-1}  + vx_{k-1}*dt
      y_k  = y_{k-1}  + vy_{k-1}*dt
      vx_k = vx_{k-1}
      vy_k = vy_{k-1}

    Measurement model:
      z_k = [x_k, y_k]^T  (GPS position)

    Process noise:
      We model unknown accelerations as "white noise acceleration".
      The scalar q_accel controls how aggressive/turny the UAV can be.
    """

    def __init__(self, x0, P0, R, q_accel=1.0):

        # Store state estimate and covariance (posterior = AFTER update)

        self.x = _as_col(x0)                   # x_k^+ : best estimate after measurement
        self.P = np.array(P0, dtype=float)     # P_k^+ : uncertainty of x_k^+


        # Store measurement noise covariance (sensor uncertainty)
        # R tells the filter how much to trust measurements.
        # Larger R => trust measurements less.

        self.R = np.array(R, dtype=float)


        # Basic dimension checks (prevents silent math bugs)

        self.n = self.x.shape[0]               # number of states
        assert self.n == 4, "State must be [x,y,vx,vy] so dimension is 4."
        assert self.P.shape == (4, 4), "P0 must be 4x4."
        assert self.R.shape == (2, 2), "R must be 2x2 for [x,y] measurements."


        # Process noise tuning parameter
        # This is the tunning nobe

        self.q_accel = float(q_accel)


        # Measurement matrix H for z = Hx
        # z = [x, y] so H picks out x and y from the state

        self.H = np.zeros((2, 4))
        self.H[0, 0] = 1.0    # measure x
        self.H[1, 1] = 1.0    # measure y


        # Save histories to be able to plot and show results in your video

        self.x_post_hist = [self.x.copy()]     # posterior state history
        self.P_post_hist = [self.P.copy()]     # posterior covariance history
        self.x_prior_hist = []                 # prior state history (prediction only)
        self.P_prior_hist = []                 # prior covariance history
        self.K_hist = []                       # Kalman gains
        self.innov_hist = []                   # innovations (measurement residuals)


    #  math healper function: build F (state transition) and Q (process noise)
            # soucred from | https://aleksandarhaber.com/extended-kalman-filter-tutorial-with-example-and-disciplined-python-codes-part-ii-python-codes/


    def _F(self, dt):
        """
        Build the state transition matrix F for constant velocity.

        WHY:
          The prediction step uses: x^- = F x^+
          This is the discrete-time kinematics model.
        """
        F = np.eye(4)
        F[0, 2] = dt   # x depends on vx
        F[1, 3] = dt   # y depends on vy
        return F

    def _Q(self, dt):
        """
        Build the process noise covariance Q.

        WHY:
          Even if we assume constant velocity, real UAVs accelerate,
          turn, and are affected by wind. Q models this uncertainty.

          Using a "white acceleration noise" model is standard:
            q_accel controls how much random acceleration we allow.

        Continuous intuition:
          acceleration is random noise -> velocity drifts -> position drifts more

        Discrete-time blocks (1D):
          Q1 = q * [[dt^3/3, dt^2/2],
                    [dt^2/2, dt]]

        We apply that to x/vx and y/vy blocks.
        """
        q = self.q_accel
        dt2 = dt * dt
        dt3 = dt2 * dt

        Q1 = np.array([[dt3 / 3.0, dt2 / 2.0],
                       [dt2 / 2.0, dt]], dtype=float) * q

        # Build 4x4 Q for [x, y, vx, vy]
        # x,vx block = Q1
        # y,vy block = Q1
        Q = np.array([
            [Q1[0, 0], 0.0,      Q1[0, 1], 0.0],
            [0.0,      Q1[0, 0], 0.0,      Q1[0, 1]],
            [Q1[1, 0], 0.0,      Q1[1, 1], 0.0],
            [0.0,      Q1[1, 0], 0.0,      Q1[1, 1]],
        ], dtype=float)

        return Q


    # EKF steps: predict() then update()/appened to state vector list


    def predict(self, dt):
        """
        Prediction step (time update).

        Equations:
          x_k^- = F x_{k-1}^+
          P_k^- = F P_{k-1}^+ F^T + Q

        WHAT it does:
          - uses the motion model to predict where the UAV will be next
          - increases uncertainty because time passes and model is imperfect
        """
        dt = float(dt)

        # If dt is 0 or negative due to messy timestamps, protect the math
        if dt <= 0:
            dt = 1e-6

        # 1) Build model matrices for this dt
        F = self._F(dt)
        Q = self._Q(dt)

        # 2) Predict next state (prior estimate)
        x_prior = F @ self.x

        # 3) Predict next covariance (prior uncertainty)
        P_prior = F @ self.P @ F.T + Q

        # 4) Save priors (useful for debugging/plots)
        self.x_prior_hist.append(x_prior.copy())
        self.P_prior_hist.append(P_prior.copy())

        # 5) Store priors as the current internal state, ready for update()
        self.x = x_prior
        self.P = P_prior

        return x_prior, P_prior

    def update(self, z):
        """
        Measurement update (correction step).

        Equations:
          innov = z - H x^-
          S     = H P^- H^T + R
          K     = P^- H^T S^{-1}
          x^+   = x^- + K innov
          P^+   = (I - K H) P^- (Joseph form)

        WHAT it does:
          - compares predicted measurement vs actual measurement
          - uses that difference (innovation) to correct the state
          - reduces uncertainty in the measured directions
        """
        # Ensure measurement is [x_meas, y_meas]^T as a (2,1) column
        z = _as_col(z)
        assert z.shape == (2, 1), "Measurement must be [x_meas, y_meas]."

        H = self.H

        # 1) Innovation: measurement residual
        innov = z - (H @ self.x)

        # 2) Innovation covariance: how uncertain we expect innov to be
        S = H @ self.P @ H.T + self.R

        # 3) Kalman Gain: how much to trust measurement vs prediction
        K = self.P @ H.T @ np.linalg.inv(S)

        # 4) Correct the state estimate using the innovation
        self.x = self.x + K @ innov

        # 5) Correct covariance (Joseph stabilized form)
        # WHY Joseph form:
        #   keeps P symmetric and numerically stable, which matters in real code
        I = np.eye(self.n)
        I_minus_KH = I - K @ H
        self.P = I_minus_KH @ self.P @ I_minus_KH.T + K @ self.R @ K.T

        # 6) Save histories for plots and video explanation
        self.K_hist.append(K.copy())
        self.innov_hist.append(innov.copy())
        self.x_post_hist.append(self.x.copy())
        self.P_post_hist.append(self.P.copy())

        return self.x, self.P, innov, K


    # Output helper function (so your driver script stays clean)
   

    def get_posterior_states(self):
        """
        Return posterior states as array shape (N,4) with columns:
          [x, y, vx, vy]
        """
        return np.hstack(self.x_post_hist).T

    def get_prior_states(self):
        """
        Return prior (predicted) states as array shape (N-1,4).
        Priors start at step 1 because step 0 is the initial condition.
        """
        if len(self.x_prior_hist) == 0:
            return np.empty((0, 4))
        return np.hstack(self.x_prior_hist).T