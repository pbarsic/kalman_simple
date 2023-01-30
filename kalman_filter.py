import copy
import logging
import numpy as np


class Kalman2D:
    def __init__(
        self,
        initial_measurement: np.ndarray,
        estimated_measurement_error: np.ndarray = np.ones(4),
    ):
        self.KG = np.eye(4)
        self.H = np.eye(4)
        self.C = np.eye(4)
        self._load_measurement_error(estimated_measurement_error)

        self.time = initial_measurement[0]
        self.state = np.concatenate([initial_measurement[1:], np.zeros(2)])
        self._load_process_covariance(np.eye(4))
        self._load_data(initial_measurement)
        self.state = self.input_state

    def _load_data(self, new_measurement):
        # load new X_m, t
        # check for shape of new data
        returnval = False
        if new_measurement.ndim == 1:
            if new_measurement.shape[0] == 3:
                # it comes in as [t, x, y]
                self.tdelta = new_measurement[0] - self.time
                # logging.debug(
                #     f"Kalman2d {self.tdelta} {self.time} {new_measurement[0]}"
                # )
                # initialize the velocities with ones
                if abs(new_measurement[0] - self.time) < 1e-6:
                    est_velocity = np.zeros(2)
                else:
                    est_velocity = (new_measurement[1:] - self.state[:2]) / (
                        new_measurement[0] - self.time
                    )
                self.time = new_measurement[0]
                self.input_state = np.concatenate([new_measurement[1:], est_velocity])
                # ignoring zk, measurement noise
                self.measurement = np.matmul(self.C, self.input_state)
                returnval = True
        if not returnval:
            logging.error(
                "Input data incorrectly formed, expected ndarray of shape (3,),"
                f" received {type(new_measurement)} {new_measurement.shape}"
            )
        return returnval

    def _predict(self):
        dt = self.tdelta
        self.A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        # logging.debug(f"Kalman2D._predict {self.state}")
        # predict new X_k
        self._predict_state()
        # predict new P_k
        self._predict_process_covariance()

    def _predict_state(self):
        # ignoring B, uk, and wk
        self.predicted_state = np.matmul(self.A, self.state)

    def _predict_process_covariance(self):
        # ignoring Qk
        # logging.debug(
        #     "Kalman2D._predict_process_covariance"
        #     f"\n{self.process_covariance}\n{self.A}\n{self.A.T}"
        #     f"\n{np.matmul(self.A, self.process_covariance)}"
        # )
        self.predicted_process_covariance = np.matmul(
            np.matmul(self.A, self.process_covariance), self.A.T
        )

    def _compute_gain(self):
        # compute Kalman gain
        Sk = (
            np.matmul(np.matmul(self.H, self.predicted_process_covariance), self.H.T)
            + self.R
        )
        # logging.debug(
        #     f"Kalman2D._compute_gain {self.predicted_process_covariance} {Sk}"
        # )
        self.KG = np.matmul(
            np.matmul(self.predicted_process_covariance, self.H), np.linalg.inv(Sk)
        )
        # logging.debug(
        #     f"compute gain\n{self.predicted_process_covariance}\n{Sk}\n{np.linalg.inv(Sk)}"
        #     f"\n{np.matmul(Sk, np.linalg.inv(Sk))}"
        #     f"\nKG num \n{np.matmul(self.predicted_process_covariance, self.H)}"
        #     f"\nKG \n{self.KG}"
        # )

    def _load_measurement_error(self, measurement_error: np.ndarray) -> bool:
        return_value = False
        if measurement_error.ndim == 1:
            if measurement_error.shape[0] == 4:
                self.R = np.eye(4)
                for ii in range(4):
                    self.R[ii, ii] = measurement_error[ii]

                return_value = True
        if not return_value:
            logging.error(
                f"Estimated measurement error must have 4 values, you provided {measurement_error.shape}"
            )
        return return_value

    def _load_process_covariance(self, pc: np.ndarray) -> bool:
        return_value = False
        if pc.ndim == 2:
            if pc.shape[0] == 4 and pc.shape[1] == 4:
                self.process_covariance = copy.copy(pc)
                return_value = True
        if not return_value:
            logging.error(
                f"Process covariance must be of shape [4,4], you provided {pc.shape}"
            )
        return return_value

    def _compute_next_state(self):
        innovation = self.measurement - np.matmul(self.H, self.predicted_state)
        self.next_state = self.predicted_state + np.matmul(self.KG, innovation)
        # logging.debug(
        #     f"Kalman2D._compute_next_state \ninnovation\n{innovation}"
        #     f"\nKG\n{self.KG}"
        #     f"\n{np.matmul(self.KG,innovation)}"
        #     f"\nnext_state {self.next_state}"
        # )

    def _compute_next_process_covariance(self):
        self.next_process_covariance = np.matmul(
            np.eye(4) - np.matmul(self.KG, self.H), self.predicted_process_covariance
        )

    def update(self, new_measurement):
        returnvalue = self._load_data(new_measurement)
        if returnvalue:
            self._predict()
            self._compute_gain()
            self._compute_next_state()
            self._compute_next_process_covariance()
            # prepare for the next iteration
            self.state = self.next_state
            self.process_covariance = self.next_process_covariance
        return returnvalue

    def get_time(self) -> np.ndarray:
        return self.time

    def get_position(self) -> np.ndarray:
        return self.state[:2]

    def get_velocity(self) -> np.ndarray:
        return self.state[2:]
