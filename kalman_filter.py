import copy
import logging
import numpy as np


class Kalman2D:
    def __init__(
        self, initial_measurement: np.ndarray, initial_pc: np.ndarray = np.eye(4)
    ):
        self.new_time = initial_measurement[0]
        self.new_state = None
        logging.info(initial_pc)
        self._load_process_covariance(initial_pc)
        self._load_data(initial_measurement)
        self.state = self.new_state

        logging.info(f"start time: {self.new_time}")

        self.KG = np.eye(2)
        self.H = np.eye(2)
        self.C = np.eye(2)
        self.R = np.zeros([2, 2])

    def _load_data(self, new_measurement):
        # load new X_m, t
        # check for shape of new data
        returnval = False
        if new_measurement.ndim == 1:
            if new_measurement.shape[0] == 3:
                # it comes in as [t, x, y]
                self.tdelta = new_measurement[0] - self.new_time
                logging.debug(
                    f"Kalman2d {self.tdelta} {self.new_time} {new_measurement[0]}"
                )
                self.new_time = new_measurement[0]
                # initialize the velocities with ones
                self.new_state = np.concatenate([new_measurement[1:], np.ones(2)])
                returnval = True
        if not returnval:
            logging.error(
                "Input data incorrectly formed, expected ndarray of shape (3,),"
                f" received {type(new_measurement)} {new_measurement.shape}"
            )
        return returnval

    def _predict(self):
        # predict new X_k
        dt = self.tdelta
        self.A = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]])
        # ignoring B, U, and w
        # predict new X_k
        self.next_state = np.matmul(self.A, self.state)
        # predict new P_k
        logging.debug(
            f"Kalman2D._predict process_covariance {self.A.shape} {self.process_covariance.shape}"
        )
        self.next_P = np.matmul(np.matmul(self.A, self.process_covariance), self.A.T)

    def _compute_gain(self):
        # compute Kalman gain
        Sk = np.matmul(np.matmul(self.H, self.next_P), self.H.T) + self.R
        self.KG = np.matmul(np.matmul(self.next_P, self.H), np.linalg.inv(Sk))

    def _compute_new_state(self):
        # compute next predicted state
        pass

    def _compute_new_process_covariance(self):
        # compute next process covariance
        pass

    def _load_process_covariance(self, pc: np.array) -> bool:
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

    def update(self, new_measurement):
        returnvalue = self._load_data(new_measurement)
        if returnvalue:
            self._predict()
            self._compute_gain()
            self._compute_new_state()
            self._compute_new_process_covariance()
            # state should be updated now
        return returnvalue

    def get_position(self) -> np.ndarray:
        return self.state[:2]

    def get_velocity(self) -> np.ndarray:
        return self.state[2:]


if __name__ == "__main__":
    pass
