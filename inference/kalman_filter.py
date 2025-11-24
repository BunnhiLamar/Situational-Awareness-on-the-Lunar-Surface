import numpy as np
from typing import Tuple, Optional, List
class KalmanFilter:
    """
    Measurement vector: [xmin, ymin, xmax, ymax, avgdepth]
    State vector: [xmin, ymin, xmax, ymax, avgdepth, vx, vy]
    (vx, vy are velos of the bounding box center)
    """
    def __init__(self, initial_bbox: np.ndarray,
                 dt: float = 0.0001,
                 process_noise_scale: float = 0.01,
                 measurement_noise_scale: float = 1.0):
        # Initialize state vector [xmin, ymin, xmax, ymax, avgdepth, vx, vy]
        self.state = np.zeros((7, 1))
        self.state[:5, 0] = initial_bbox
        self.state[5:7, 0] = 0 # init velocities

        self.dt = dt
        # State transition matrix (7x7)

        self.F = np.eye(7)
        self.F[0, 5] = dt  # xmin += vx * dt
        self.F[1, 6] = dt  # ymin += vy * dt
        self.F[2, 5] = dt  # xmax += vx * dt
        self.F[3, 6] = dt  # ymax += vy * dt
        # Measurement matrix (5x7) - we only observe the bbox and depth, not velocities
        self.H = np.zeros((5, 7))
        self.H[:5, :5] = np.eye(5)
        # Initialize covariance matrix
        self.P = np.eye(7)
        self.P[:5, :5] *= 10  # Higher uncertainty for position
        self.P[5:, 5:] *= 100  # Even higher uncertainty for initial velocities

        # Process noise covariance (7x7)
        self.Q = np.eye(7)
        # Position noise
        self.Q[0, 0] = process_noise_scale * 1  # xmin
        self.Q[1, 1] = process_noise_scale * 1  # ymin
        self.Q[2, 2] = process_noise_scale * 1  # xmax
        self.Q[3, 3] = process_noise_scale * 1  # ymax
        self.Q[4, 4] = process_noise_scale * 0.1  # depth (typically more stable)
        # Velocity noise (allowing for acceleration)
        self.Q[5, 5] = process_noise_scale * 10  # vx
        self.Q[6, 6] = process_noise_scale * 10  # vy

        # Measurement noise covariance (5x5)
        self.R = np.eye(5)
        self.R[0, 0] = measurement_noise_scale * 1  # xmin noise
        self.R[1, 1] = measurement_noise_scale * 1  # ymin noise
        self.R[2, 2] = measurement_noise_scale * 1  # xmax noise
        self.R[3, 3] = measurement_noise_scale * 1  # ymax noise
        self.R[4, 4] = measurement_noise_scale * 0.5  # depth noise

    def predict(self) -> np.ndarray:
        """
        Prediction step: Estimate next state based on motion model.
        
        Returns:
            Predicted state vector
        """
        # Predict state
        self.state = self.F @ self.state
        
        # Predict covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.state.copy()
    def update(self, measurement: np.ndarray) -> np.ndarray:
        """
        Update step: Correct prediction based on new measurement.
        
        Args:
            measurement: Observed [xmin, ymin, xmax, ymax, avgdepth]
            
        Returns:
            Updated state vector
        """
        z = np.array(measurement).reshape((5, 1))
        # z = measurement
        # Calculate Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Calculate innovation (measurement residual)
        y = z - self.H @ self.state
        
        # Update state
        self.state = self.state + K @ y
        
        # Update covariance
        I = np.eye(7)
        self.P = (I - K @ self.H) @ self.P
        
        return self.state.copy()
    def get_bbox(self) -> Tuple[float, float, float, float, float]:
        """
        Get current bounding box from state.
        
        Returns:
            (xmin, ymin, xmax, ymax, avgdepth)
        """
        return tuple(self.state[:5, 0])
    
    def get_velocity(self) -> Tuple[float, float]:
        """
        Get current velocity from state.
        
        Returns:
            (vx, vy)
        """
        return tuple(self.state[5:7, 0])
    
    def get_center(self) -> Tuple[float, float]:
        """
        Get bounding box center.
        
        Returns:
            (cx, cy)
        """
        xmin, ymin, xmax, ymax = self.state[:4, 0]
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        return cx, cy