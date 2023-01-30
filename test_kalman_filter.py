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
        self.tvals = np.arange(1, 11, self.dt)
        self.x0 = 200
        self.xs = -0.8
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
        estimated_measurement_error = np.array(
            [
                5,
                4,
                0,
                0,
            ]
        )
        kf = kalman_filter.Kalman2D(estimated_measurement_error)
        self.assertEqual(kf.get_time(), 0)
        self.assertTupleEqual(tuple(kf.get_position().tolist()), (0, 0))
        self.assertTupleEqual(tuple(kf.get_velocity().tolist()), (0, 0))
        self.assertEqual(kf.num_points, 0)

    def test_load_data(self):
        logging.info("test_load_data")
        initial_datum = self.clean[0, :]
        kf = kalman_filter.Kalman2D()

        self.assertEqual(kf.state[0], 0)
        self.assertEqual(kf.state[1], 0)
        self.assertEqual(kf.state[2], 0)
        self.assertEqual(kf.state[3], 0)

        kf._load_data(initial_datum)
        self.assertEqual(kf.tdelta, 0)
        self.assertEqual(kf.time, initial_datum[0])
        self.assertEqual(kf.input_state[0], initial_datum[1])
        self.assertEqual(kf.input_state[1], initial_datum[2])
        self.assertEqual(kf.input_state[2], 0)
        self.assertEqual(kf.input_state[3], 0)
        self.assertEqual(kf.num_points, 1)
        # since this test just loads but doesn't update, expect state to stay at the first value
        self.assertEqual(kf.state[0], initial_datum[1])
        self.assertEqual(kf.state[1], initial_datum[2])
        self.assertEqual(kf.state[2], 0)
        self.assertEqual(kf.state[3], 0)

        datum = self.clean[1, :]
        self.assertTrue(kf._load_data(datum))
        self.assertEqual(kf.tdelta, self.dt)
        self.assertEqual(kf.time, datum[0])
        self.assertEqual(kf.input_state[0], datum[1])
        self.assertEqual(kf.input_state[1], datum[2])
        self.assertAlmostEqual(kf.input_state[2], self.xs, 5)
        self.assertAlmostEqual(kf.input_state[3], self.ys, 5)
        self.assertEqual(kf.num_points, 2)
        # since this test just loads but doesn't update, expect state to stay at the first value
        self.assertEqual(kf.state[0], initial_datum[1])
        self.assertEqual(kf.state[1], initial_datum[2])
        self.assertEqual(kf.state[2], 0)
        self.assertEqual(kf.state[3], 0)

    def test_predict(self):
        logging.info("test_predict")
        estimated_measurement_error = np.ones(4) * 1e-6
        kf = kalman_filter.Kalman2D(estimated_measurement_error)
        # insert some fake values
        initial_datum = np.array([0, 500, 400])
        kf._load_data(initial_datum)
        kf.tdelta = 5
        kf.process_covariance = np.array(
            [[1, 0, 2, 0], [0, 2, 0, 3], [0, 0, 1, 0], [0, 0, 0, 9]]
        )
        # velocity initialized to 0, so expect same as initial state
        expected_predicted_state = np.array([500, 400, 0, 0])
        expected_pc = np.array(
            [[36, 0, 7, 0], [0, 242, 0, 48], [5, 0, 1, 0], [0, 45, 0, 9]]
        )
        datum = np.array([5, 506, 404])
        self.assertTrue(kf._load_data(datum))
        kf._predict()
        # logging.debug(f"state\n{kf.predicted_state}\n{expected_predicted_state}")
        self.assertTrue(all(kf.predicted_state == expected_predicted_state))
        # logging.debug(
        #     f"\n{kf.predicted_process_covariance}\n{estimated_measurement_error}"
        # )
        self.assertTrue(
            all(kf.predicted_process_covariance.ravel() == expected_pc.ravel())
        )

    def test_gain(self):
        logging.info("test_gain")
        estimated_measurement_error = np.ones(4) * 1e-8
        kf = kalman_filter.Kalman2D(estimated_measurement_error)
        # insert some fake values
        kf.tdelta = 5
        kf.process_covariance = np.array(
            [[1, 0, 2, 0], [0, 2, 0, 3], [0, 0, 1, 0], [0, 0, 0, 9]]
        )

        expected_gain = np.eye(4)
        datum = np.array([5, 506, 404])
        self.assertTrue(kf._load_data(datum))
        kf._predict()
        kf._compute_gain()
        # logging.debug(f"KG\n{kf.KG.ravel()}\n{expected_gain.ravel()}\n{all(kf.KG.ravel() - expected_gain.ravel() < 1e-6)}")

        self.assertTrue(all(abs(kf.KG.ravel() - expected_gain.ravel()) < 1e-6))

    def test_compute_next_state(self):
        logging.info("test_compute_next_state")
        kf = kalman_filter.Kalman2D()
        # fill them with fake values
        kf.measurement = np.array([1, 2, 3, 4])
        kf.predicted_state = np.array([2, -2, -2, -2])
        expected_next_state = np.array([0, 1, 2, 3])  # just do the math by hand
        kf.KG = np.array([[0, -2, 0, 1], [1, 1, 0, 0], [0, 1, 0, 0], [1, 0, 0, 1]])
        kf._compute_next_state()
        self.assertTrue(all(kf.next_state == expected_next_state))

    def test_compute_next_process_covariance(self):
        logging.info("test_compute_next_process_covariance")
        kf = kalman_filter.Kalman2D()
        # fill them with fake values
        kf.predicted_process_covariance = np.array(
            [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33], [40, 41, 42, 43]]
        )
        kf.KG = np.array([[1, 0, 0, -1], [-1, 1, 0, 0], [0, -1, 1, 0], [0, 0, -1, 1]])
        kf._compute_next_process_covariance()
        expected_next_process_covariance = np.array(
            [[40, 41, 42, 43], [10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
        )
        self.assertTrue(
            all(
                kf.next_process_covariance.ravel()
                == expected_next_process_covariance.ravel()
            )
        )

    def test_fit_clean(self):
        logging.info("test_fit_clean")
        estimated_measurement_error = np.ones(4) * 1e-3
        kf = kalman_filter.Kalman2D(estimated_measurement_error)
        desired_state = np.array([self.xvals[-1], self.yvals[-1], self.xs, self.ys])
        for ii in range(self.clean.shape[0]):
            datum = self.clean[ii, :]
            self.assertTrue(kf.update(datum))
            logging.info(f"CC {datum} {kf.state} {kf.tdelta} {desired_state}")
        for calc, expect in zip(kf.state, desired_state):
            self.assertAlmostEqual(calc, expect, 0)

    def test_fit_dirty(self):
        logging.info("test_fit_dirty")
        estimated_measurement_error = np.ones(4) * 0.1
        kf = kalman_filter.Kalman2D(estimated_measurement_error)
        desired_state = np.array([self.xvals[-1], self.yvals[-1], self.xs, self.ys])
        for ii in range(self.dirty.shape[0]):
            datum = self.dirty[ii, :]
            self.assertTrue(kf.update(datum))
            logging.info(f"DD {datum} {kf.state} {kf.tdelta} {desired_state}")
        for calc, expect in zip(kf.state, desired_state):
            self.assertAlmostEqual(calc, expect, 0)

        final_value = kf.get_position()
        self.assertTupleEqual(final_value.shape, (2,))
        self.assertAlmostEqual(desired_state[0], final_value[0], 0)
        self.assertAlmostEqual(desired_state[1], final_value[1], 0)

        final_value = kf.get_velocity()
        self.assertTupleEqual(final_value.shape, (2,))
        self.assertAlmostEqual(desired_state[2], final_value[0], 1)
        self.assertAlmostEqual(desired_state[3], final_value[1], 1)

        final_time = kf.get_time()
        self.assertEqual(final_time, self.dirty[-1, 0])


if __name__ == "__main__":
    logger_filename = anonymous_logger.logger_setup(
        "test_kalman.txt",
        log_folder=Path("/tmp/kalman/"),
        console_output=False,
        overwrite_old=True,
    )
    print(logger_filename)
    unittest.main()
