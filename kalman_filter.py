import copy
import logging
import numpy as np


class Kalman2D:
    def __init__(
        self, initial_measurement: np.ndarray, initial_pc: np.ndarray = np.eye(4)
    ):
        self.KG = np.eye(4)
        self.H = np.eye(4)
        self.C = np.eye(4)
        self.R = np.zeros([4, 4])

        self.new_time = initial_measurement[0]
        self.input_state = None
        logging.info(initial_pc)
        self._load_process_covariance(initial_pc)
        self._load_data(initial_measurement)
        self.state = self.input_state

        logging.info(f"start time: {self.new_time}")

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
                self.input_state = np.concatenate([new_measurement[1:], np.ones(2)])
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
        logging.debug(f"Kalman2D._predict {self.state}")
        # predict new X_k
        self._predict_state()
        # predict new P_k
        self._predict_process_covariance()

    def _predict_state(self):
        # ignoring B, uk, and wk
        self.predicted_state = np.matmul(self.A, self.state)

    def _predict_process_covariance(self):
        # ignoring Qk
        self.predicted_process_covariance = np.matmul(
            np.matmul(self.A, self.process_covariance), self.A.T
        )

    def _compute_gain(self):
        # compute Kalman gain
        Sk = (
            np.matmul(np.matmul(self.H, self.predicted_process_covariance), self.H.T)
            + self.R
        )
        self.KG = np.matmul(
            np.matmul(self.predicted_process_covariance, self.H), np.linalg.inv(Sk)
        )
        # logging.debug(
        #     f"compute gain\n{self.predicted_process_covariance}\n{Sk}\n{np.linalg.inv(Sk)}"
        #     f"\n{np.matmul(Sk, np.linalg.inv(Sk))}"
        #     f"\nKG num \n{np.matmul(self.predicted_process_covariance, self.H)}"
        #     f"\nKG \n{self.KG}"
        # )

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
        # logging.debug(
        #     f"Kalman2D._compute_next_state \ninnovation\n{innovation}"
        #     f"\nKG\n{self.KG}"
        #     f"\n{np.matmul(self.KG,innovation)}"
        # )
        self.next_state = self.predicted_state + np.matmul(self.KG, innovation)

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
            self.state = self.predicted_state
            self.process_covariance = self.predicted_process_covariance
        return returnvalue

    def get_position(self) -> np.ndarray:
        return self.state[:2]

    def get_velocity(self) -> np.ndarray:
        return self.state[2:]


if __name__ == "__main__":
    pass
