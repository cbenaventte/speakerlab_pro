import tempfile
import unittest
from pathlib import Path

import numpy as np

from api.pdf_generator import calc_acoustics, generate_pdf


def reference_design(**overrides):
    values = {
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
        "le": 1.8,
        "qb": 7,
        "box_type": "reflex",
        "alignment": "QB3",
        "port_diam_cm": 7,
        "num_ports": 1,
        "eg_volts": 5.66,
        "model_name": "PDF Test Driver",
        "language": "es",
    }
    values.update(overrides)
    return values


class PdfConsistencyTests(unittest.TestCase):
    def test_pdf_data_uses_canonical_simulation_results(self):
        result = calc_acoustics(reference_design())
        simulation = result["sim_data"]

        self.assertAlmostEqual(result["Vb"], simulation["vb_liters"])
        self.assertAlmostEqual(result["Fb"], simulation["fb"])
        self.assertAlmostEqual(result["F3"], simulation["f3_from_curve"])
        self.assertAlmostEqual(result["L"], simulation["L_port_cm"])
        self.assertAlmostEqual(result["portVel"], float(np.max(simulation["port_vel"])))
        self.assertAlmostEqual(simulation["eg_volts"], 5.66)
        self.assertAlmostEqual(simulation["phys"]["qb"], 7.0)

    def test_pdf_rejects_an_unworkable_port(self):
        with self.assertRaisesRegex(ValueError, "puerto inviable"):
            calc_acoustics(reference_design(port_diam_cm=0.5))

    def test_pdf_is_generated_in_both_supported_languages(self):
        with tempfile.TemporaryDirectory() as directory:
            for language in ("es", "en"):
                with self.subTest(language=language):
                    output = Path(directory) / f"speakerlab-{language}.pdf"
                    generate_pdf(reference_design(language=language), str(output))
                    self.assertGreater(output.stat().st_size, 10_000)
                    self.assertTrue(output.read_bytes().startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
