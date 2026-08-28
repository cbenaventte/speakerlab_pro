import unittest

import numpy as np

from api.acoustic_sim import simulate, tf_closed, ts_to_physical


REFERENCE_DRIVER = {
    "fs": 30,
    "vas": 100,
    "qts": 0.35,
    "qes": 0.40,
    "qms": 4.0,
    "xmax": 8,
    "sd": 500,
    "re": 6,
    "spl": 88,
    "mms": 100,
    "bl": 12,
}


class AcousticSimulationTests(unittest.TestCase):
    def setUp(self):
        self.freqs = np.logspace(np.log10(15), np.log10(800), 200)

    def test_physical_conversion_preserves_manufacturer_values(self):
        physical = ts_to_physical(REFERENCE_DRIVER)
        self.assertAlmostEqual(physical["mms"], 0.1)
        self.assertAlmostEqual(physical["bl"], 12.0)
        self.assertAlmostEqual(physical["sd"], 0.05)

    def test_qb3_reference_design(self):
        result = simulate(
            {**REFERENCE_DRIVER, "box_type": "reflex", "alignment": "QB3"},
            freqs=self.freqs,
        )
        self.assertEqual(result["box_type"], "reflex")
        self.assertAlmostEqual(result["vb_liters"], 57.5, places=1)
        self.assertAlmostEqual(result["fb"], 40.5, places=1)
        self.assertGreaterEqual(result["L_port_cm"], 1.0)
        self.assertEqual(len(result["spl"]), len(self.freqs))
        self.assertTrue(np.all(np.isfinite(result["spl"])))
        self.assertTrue(np.all(result["impedance"] > 0))

    def test_closed_reference_design_reaches_target_qtc(self):
        result = simulate(
            {**REFERENCE_DRIVER, "box_type": "closed", "qtc_target": 0.707},
            freqs=self.freqs,
        )
        expected_vb = 100 / ((0.707 / 0.35) ** 2 - 1)
        self.assertEqual(result["box_type"], "closed")
        self.assertAlmostEqual(result["qtc_real"], 0.707, places=3)
        self.assertAlmostEqual(result["vb_liters"], expected_vb, delta=0.1)
        self.assertNotIn("port_vel", result)

    def test_closed_box_rejects_impossible_qtc(self):
        physical = ts_to_physical(REFERENCE_DRIVER)
        with self.assertRaisesRegex(ValueError, "debe ser > Qts"):
            tf_closed(physical, qtc_target=0.30)


if __name__ == "__main__":
    unittest.main()
