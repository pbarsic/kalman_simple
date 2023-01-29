import kalman_filter

import logging
import numpy as np
from pathlib import Path
import unittest

import anonymous_logger


def ft(b, m, tvals):
    return b + m * tvals


class test_kalman(unittest.TestCase):
    def setUp(self) -> None:
        self.dt = 0.5
        self.tvals = np.arange(0, 10, self.dt)
        self.x0 = 200
        self.xs = -0.5
        self.y0 = 400
        self.ys = 0.75
        self.xvals = ft(self.x0, self.xs, self.tvals)
        self.yvals = ft(self.y0, self.ys, self.tvals)

        # some constant random noise
        # (generated y ydirt = np.random.randn(self.tvals.shape[0]) * 0.1 )
        self.xdirt = np.array(
            [
                -0.04063426,
                -0.09977588,
                0.07976062,
                0.16942444,
                0.02735772,
                -0.10336854,
                -0.03835327,
                -0.02324728,
                0.01218789,
                -0.00388813,
                -0.05997252,
                0.13219375,
                -0.07365002,
                -0.00198964,
                0.09416759,
                0.0471458,
                -0.0436184,
                0.00618637,
                0.04566814,
                0.10684797,
            ]
        )

        self.ydirt = np.array(
            [
                -0.08739665,
                -0.13539456,
                -0.0559583,
                -0.02373654,
                0.07077156,
                -0.02318059,
                0.01578514,
                0.09897362,
                -0.0401714,
                -0.22426971,
                -0.06951351,
                -0.01066362,
                0.12613457,
                -0.05878731,
                -0.04657987,
                -0.09753212,
                0.06538713,
                0.10698413,
                0.00241145,
                -0.11400357,
            ]
        )

        self.tdirt = np.array(
            [
                0.10484457,
                0.18007072,
                -0.02875083,
                -0.15782624,
                0.1184257,
                0.12200035,
                0.08624234,
                0.0198951,
                0.07559771,
                0.08972216,
                0.04637315,
                -0.14118951,
                0.06183095,
                0.07221524,
                0.01185314,
                -0.01923092,
                -0.06481772,
                0.02174371,
                0.11998992,
                -0.06697347,
            ]
        )

        self.clean = (
            np.concatenate([self.tvals, self.xvals, self.yvals]).reshape(3, -1).T
        )
        self.dirty = (
            self.clean
            + np.concatenate([self.tdirt, self.xdirt, self.ydirt]).reshape(3, -1).T
        )

        return super().setUp()

    def tearDown(self) -> None:
        return super().tearDown()

    def test_init(self):
        logging.info("test_init")
        initial_state = np.array([9, 50, 40])
        initial_pc = np.array([[5, 0], [0, 4]])
        kf = kalman_filter.Kalman2D(initial_state, initial_pc)
        self.assertEqual(kf.new_time, initial_state[0])
        self.assertEqual(kf.tdelta, 0)

    def Xtest_load_data(self):
        initial_state = self.clean[0, :]
        initial_pc = np.eye(4)
        kf = kalman_filter.Kalman2D(initial_state, initial_pc)
        self.assertEqual(kf.tdelta, 0)
        self.assertEqual(kf.new_time, initial_state[0])
        self.assertEqual(kf.state[0], initial_state[1])
        self.assertEqual(kf.new_state[0], initial_state[1])
        self.assertEqual(kf.state[1], initial_state[2])
        self.assertEqual(kf.new_state[1], initial_state[2])
        self.assertEqual(kf.state[2], 1)
        self.assertEqual(kf.new_state[2], 1)
        self.assertEqual(kf.state[3], 1)
        self.assertEqual(kf.new_state[3], 1)

        datum = self.clean[1, :]
        self.assertTrue(kf._load_data(datum))
        self.assertEqual(kf.tdelta, self.dt)
        self.assertEqual(kf.new_time, datum[0])
        self.assertEqual(kf.new_state[0], datum[1])
        self.assertEqual(kf.new_state[1], datum[2])
        self.assertEqual(kf.new_state[2], 1)
        self.assertEqual(kf.new_state[3], 1)
        # since this just loads but doesn't update, expect state
        # to never change
        self.assertEqual(kf.state[0], initial_state[1])
        self.assertEqual(kf.state[1], initial_state[2])
        self.assertEqual(kf.state[2], 1)
        self.assertEqual(kf.state[3], 1)

    def test_predict(self):
        initial_state = np.array([0, 500, 400])
        initial_pc = np.array([[1, 0, 2, 0], [0, 2, 0, 3], [0, 0, 1, 0], [0, 0, 0, 9]])
        kf = kalman_filter.Kalman2D(initial_state, initial_pc)
        # insert some fake values
        kf.tdelta = 5
        expected_next_state = np.array([505, 405, 1, 1])
        expected_pc = np.array(
            [[36, 0, 7, 0], [0, 242, 0, 48], [5, 0, 1, 0], [0, 45, 0, 9]]
        )
        datum = np.array([5, 506, 404])
        self.assertTrue(kf._load_data(datum))
        kf._predict()
        self.assertTrue(all(kf.next_state == expected_next_state))
        logging.debug(f"\n{kf.next_P}\n{initial_pc}")
        self.assertTrue(all(kf.next_P.ravel() == expected_pc.ravel()))

    def Xtest_gain(self):
        initial_state = np.array([0, 500, 400])
        initial_pc = np.array([[1, 0, 2, 0], [0, 2, 0, 3], [0, 0, 1, 0], [0, 0, 0, 9]])
        kf = kalman_filter.Kalman2D(initial_state, initial_pc)
        # insert some fake values
        kf.tdelta = 5
        expected_gain = np.eye(4)
        datum = np.array([5, 506, 404])
        self.assertTrue(kf._load_data(datum))
        kf._predict()
        kf._compute_gain()
        # logging.debug(f"KG\n{kf.KG.ravel()}\n{expected_gain.ravel()}\n{all(kf.KG.ravel() - expected_gain.ravel() < 1e-6)}")

        self.assertTrue(all(abs(kf.KG.ravel() - expected_gain.ravel()) < 1e-6))

        

    def SKIP_test_fit_dirty(self):
        pass


if __name__ == "__main__":
    logger_filename = anonymous_logger.logger_setup(
        "test_kalman.txt",
        log_folder=Path("/tmp/kalman/"),
        console_output=False,
        overwrite_old=True,
    )
    print(logger_filename)
    unittest.main()
