import copy
import numpy as np


class Kalman2D:
    def __init__(
        self, initial_measurement: np.ndarray, initial_pc: np.ndarray = np.eye(2)
    ):
        self.new_time = initial_measurement[0]
        self.load_data(initial_measurement)

        logging.info(f"start time: {self.new_time}")

        self.KG = np.eye(2)

    def load_data(self, new_measurement):
        # load new X_m, t
        # check for shape of new data
        returnval = False
        if new_measurement.ndim == 1:
            if new_measurement.shape[0] == 3:
                # it comes in as [t, x, y]
                self.tdelta = self.new_time - new_measurement[0]
                self.new_time = new_measurement[0]
                self.new_state = np.concatenate([new_measurement[1:], np.array[0, 0]])
                returnval = True
        return returnval

    def predict(self):
        # predict new X_k
        # predict new P_k
        pass

    def compute_gain(self):
        # compute Kalman gain
        pass

    def compute_new_state(self):
        # compute next predicted state
        pass

    def compute_new_process_covariance(self):
        # compute next process covariance
        pass

    def load_process_covariance(self, pc: np.array) -> bool:
        return_value = False
        if pc.ndim == 2:
            if pc.shape[0] == 2 and pc.shape[1] == 2:
                self.process_covariance = copy.copy(pc)
                return_value = True
        if return_value == False:
            logging.error(
                f"Process covariance must be of shape [2,2], you provided {pc.shape}"
            )
        return return_value

    def update(self, new_measurement):
        if self.load_data(new_measurement):
            self.predict()
            self.compute_gain()
            self.compute_new_state()
            self.compute_new_process_covariance()
            # state should be updated now


if __name__ == "__main__":
    pass
