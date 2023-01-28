import kalman_filter

import logging
import numpy as np
from pathlib import Path
import unittest

import anonymous_logger


class test_kalman(unittest.TestCase):
    def setUp(self) -> None:
        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_init(self):
        logging.info("test_init")
        initial_state = np.array([9, 50, 40])
        initial_pc = np.array([[5, 0], [0, 4]])
        kf = kalman.Kalman2D(initial_state, initial_pc)
        self.assertEqual(kf.start_time, initial_state[0])
        self.assertEqual(kf.tdelta, 0)



if __name__ == "__main__":
    logger_filename = anonymous_logger.logger_setup(
        "test_kalman.txt",
        log_folder=Path("/tmp/kalman/"),
        console_output=False,
        overwrite_old=True,
    )
    print(logger_filename)
    unittest.main()
