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
        physical = ts_to_physical({**REFERENCE_DRIVER, "le": 1.8})
        self.assertAlmostEqual(physical["mms"], 0.1)
        self.assertAlmostEqual(physical["bl"], 12.0)
        self.assertAlmostEqual(physical["sd"], 0.05)
        self.assertAlmostEqual(physical["le"], 1.8e-3)

    def test_voice_coil_inductance_changes_high_frequency_impedance(self):
        freqs = np.array([500.0, 1000.0])
        low_le = simulate(
            {**REFERENCE_DRIVER, "le": 0.1, "box_type": "closed", "qtc_target": 0.707},
            freqs=freqs,
        )
        high_le = simulate(
            {**REFERENCE_DRIVER, "le": 5.0, "box_type": "closed", "qtc_target": 0.707},
            freqs=freqs,
        )
        self.assertGreater(high_le["impedance"][0], low_le["impedance"][0])

    def test_drive_voltage_scales_spl_excursion_and_power_consistently(self):
        low = simulate(
            {**REFERENCE_DRIVER, "box_type": "closed", "qtc_target": 0.707},
            freqs=self.freqs,
            eg_volts=2.83,
        )
        high = simulate(
            {**REFERENCE_DRIVER, "box_type": "closed", "qtc_target": 0.707},
            freqs=self.freqs,
            eg_volts=5.66,
        )
        self.assertAlmostEqual(high["sens_band"] - low["sens_band"], 6.0206, places=3)
        self.assertAlmostEqual(high["excursion"][50] / low["excursion"][50], 2.0, places=3)
        self.assertAlmostEqual(high["input_power_w"] / low["input_power_w"], 4.0, places=3)

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

    def test_unworkable_port_is_reported_without_changing_its_length(self):
        result = simulate(
            {
                **REFERENCE_DRIVER,
                "box_type": "reflex",
                "alignment": "QB3",
                "port_diam_cm": 0.5,
            },
            freqs=self.freqs,
        )
        self.assertFalse(result["port_feasible"])
        self.assertLess(result["L_port_cm"], 1.0)

    def test_reference_band_falls_back_for_low_frequency_ranges(self):
        freqs = np.logspace(np.log10(10), np.log10(100), 100)
        result = simulate(
            {**REFERENCE_DRIVER, "box_type": "closed", "qtc_target": 0.707},
            freqs=freqs,
        )
        self.assertTrue(np.isfinite(result["sens_band"]))

    def test_reflex_rejects_qts_outside_alignment_tables(self):
        with self.assertRaisesRegex(ValueError, "rango 0.20–0.50"):
            simulate(
                {**REFERENCE_DRIVER, "qts": 0.7, "qes": 0.8, "box_type": "reflex"},
                freqs=self.freqs,
            )

    def test_qb_changes_reflex_losses_and_response(self):
        lossy = simulate(
            {**REFERENCE_DRIVER, "box_type": "reflex", "alignment": "QB3", "qb": 3.0},
            freqs=self.freqs,
        )
        low_loss = simulate(
            {**REFERENCE_DRIVER, "box_type": "reflex", "alignment": "QB3", "qb": 20.0},
            freqs=self.freqs,
        )
        self.assertFalse(np.allclose(lossy["spl"], low_loss["spl"]))
        self.assertFalse(np.allclose(lossy["impedance"], low_loss["impedance"]))

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
