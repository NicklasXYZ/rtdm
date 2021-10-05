# ------------------------------------------------------------------------------#
#                     Author     : Nicklas Sindlev Andersen                    #
#                     Website    : Nicklas.xyz                                 #
#                     Github     : github.com/NicklasXYZ                       #
# ------------------------------------------------------------------------------#
#                                                                              #
# ------------------------------------------------------------------------------#
#               Import packages from the python standard library               #
# ------------------------------------------------------------------------------#
import logging

import numpy as np
# ------------------------------------------------------------------------------#
#                          Import local libraries/code                         #
# ------------------------------------------------------------------------------#
#
# ------------------------------------------------------------------------------#
#                      Import third-party libraries: Others                    #
# ------------------------------------------------------------------------------#
from filterpy.kalman import KalmanFilter

# ------------------------------------------------------------------------------#
#                         GLOBAL SETTINGS AND VARIABLES                        #
# ------------------------------------------------------------------------------#
logging.basicConfig(level=logging.DEBUG)


# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
# ------------------------------------------------------------------------------#
# Constant velocity Kalman filter (applicable if only (latitude, longitude)
# coordinates are known)
class ConstantVelocityKlamanFilter:
    def __init__(self, noise=10.0):
        # Initialize parameters for the Kalman filter
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.dt = 1.0
        self.unit_scaler = 0.001
        self.x_state = np.array([[0.0, 0.0, 0.0, 0.0]]).T
        self.noise = noise
        self.trillion = 1000.0 * 1000.0 * 1000.0 * 1000.0
        # State transition matrix (assuming constant velocity model)
        self.kf.F = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],]
        )
        self.kf.F[0][2] = self.unit_scaler * self.dt
        self.kf.F[1][3] = self.unit_scaler * self.dt
        # Measurement matrix (assuming we can only measure the coordinates)
        self.kf.H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],])
        # State covariance matrix
        self.kf.P *= self.trillion
        # Process uncertainty
        self.pos = 0.000001
        self.kf.Q = np.array(
            [
                [self.pos, 0.0, 0.0, 0.0],
                [0.0, self.pos, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        # State uncertainty
        self.kf.R = np.eye(2) * self.pos * self.noise

    # Method: Used to predict and update the next state for a bounding box
    def predict_and_update(self, z, dt):
        _z = z.copy()
        _z[0] = _z[0] * 1000.0
        _z[1] = _z[1] * 1000.0
        self.kf.x = self.x_state
        self.dt = dt
        # Predict
        self.kf.predict()
        # State transition matrix (assuming constant velocity model)
        self.kf.F[0][2] = self.unit_scaler * self.dt
        self.kf.F[1][3] = self.unit_scaler * self.dt
        # Update
        # self.kf.update(z)
        self.kf.update(_z)
        # Get current state and convert to integers for pixel coordinates
        self.x_state = self.kf.x
        _x = self.kf.x.copy()
        _x[0][0] = _x[0][0] / 1000.0
        _x[1][0] = _x[1][0] / 1000.0
        return _x
