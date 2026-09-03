import asyncio
import unittest

from fastapi import HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from api.index import (
    AlignmentRequest,
    DriverParams,
    PDFRequest,
    SimulateRequest,
    api_simulate,
    frontend_config,
    get_alignments,
    get_speakers,
    health,
)


def reference_driver(**overrides):
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
    }
    values.update(overrides)
    return DriverParams(**values)


class ApiContractTests(unittest.TestCase):
    def test_health_reports_free_access(self):
        response = asyncio.run(health())
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["access"], "free")

    def test_config_enables_free_pdf(self):
        response = asyncio.run(frontend_config())
        self.assertEqual(response["access"], "free")
        self.assertTrue(response["pdf_enabled"])

    def test_simulation_and_pdf_accept_supported_languages(self):
        driver = reference_driver()
        self.assertEqual(SimulateRequest(driver=driver, language="en").language, "en")
        self.assertEqual(PDFRequest(driver=driver, language="en").language, "en")
        with self.assertRaises(ValidationError):
            PDFRequest(driver=driver, language="fr")

    def test_canonical_speaker_database_is_available(self):
        speakers = asyncio.run(get_speakers())
        self.assertEqual(len(speakers), 15)
        identities = {(item["manufacturer"], item["model_name"]) for item in speakers}
        self.assertEqual(len(identities), len(speakers))

    def test_alignment_endpoint_uses_reference_engine(self):
        response = asyncio.run(get_alignments(AlignmentRequest(fs=30, vas=100, qts=0.35)))
        self.assertEqual(
            {key: response["alignments"]["B4"][key] for key in ("vb", "fb", "f3_table")},
            {"vb": 104.2, "fb": 34.8, "f3_table": 34.8},
        )
        self.assertGreater(response["alignments"]["B4"]["f3"], 0)
        self.assertEqual(response["closed"]["qtc"], 0.707)
        self.assertGreater(response["closed"]["vb"], 0)

    def test_alignment_endpoint_keeps_closed_design_outside_reflex_table_range(self):
        response = asyncio.run(get_alignments(AlignmentRequest(fs=30, vas=100, qts=0.7)))
        self.assertFalse(response["reflex_supported"])
        self.assertEqual(response["alignments"], {})
        self.assertGreater(response["closed"]["vb"], 0)

    def test_request_limits_reject_excessive_work(self):
        with self.assertRaises(ValidationError):
            SimulateRequest(driver=reference_driver(), freq_points=5001)
        with self.assertRaises(ValidationError):
            SimulateRequest(driver=reference_driver(), eg_volts=0)

    def test_driver_rejects_invalid_physical_parameters(self):
        invalid_cases = [
            {"qes": 0.30},
            {"qms": 0.20},
            {"sd": 0},
            {"xmax": -1},
            {"re": 0},
            {"spl": 140},
            {"port_diam_cm": 0},
            {"num_ports": 9},
            {"material_mm": 3},
            {"qb": 2.0},
            {"qb": 31.0},
        ]
        for values in invalid_cases:
            with self.subTest(values=values), self.assertRaises(ValidationError):
                reference_driver(**values)

    def test_closed_box_requires_qtc_above_qts(self):
        with self.assertRaisesRegex(ValidationError, "Qtc objetivo debe ser mayor"):
            reference_driver(box_type="closed", qts=0.8, qes=0.9, qtc_target=0.707)

    def test_frequency_range_must_be_ascending(self):
        request = SimulateRequest(
            driver=reference_driver(), freq_min=500, freq_max=100, freq_points=50
        )
        with self.assertRaises(HTTPException) as raised:
            asyncio.run(api_simulate(request))
        self.assertEqual(raised.exception.status_code, 422)

    def test_simulation_response_contract(self):
        request = SimulateRequest(driver=reference_driver(), freq_points=100)
        response = asyncio.run(api_simulate(request))
        self.assertEqual(len(response["freqs"]), 100)
        self.assertEqual(len(response["spl"]), 100)
        self.assertEqual(response["metrics"]["box_type"], "reflex")
        self.assertEqual(response["metrics"]["alignment"], "QB3")
        self.assertAlmostEqual(response["metrics"]["simulation_voltage"], 2.83)
        self.assertGreater(response["metrics"]["input_power_w"], 0)
        self.assertEqual(response["warnings"], [])
        jsonable_encoder(response)

    def test_rectangular_port_response_is_json_serializable(self):
        request = SimulateRequest(
            driver=reference_driver(
                port_type="slot",
                slot_w_cm=10,
                slot_h_cm=5,
                num_ports=1,
            ),
            freq_points=100,
        )
        response = asyncio.run(api_simulate(request))
        encoded = jsonable_encoder(response)

        self.assertIs(type(encoded["metrics"]["port_feasible"]), bool)
        self.assertGreater(encoded["metrics"]["L_port_cm"], 0)
        self.assertAlmostEqual(encoded["metrics"]["sp_cm2"], 50.0)
        self.assertEqual(len(encoded["port_vel"]), 100)


if __name__ == "__main__":
    unittest.main()
