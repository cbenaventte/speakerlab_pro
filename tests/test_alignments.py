import unittest

from api.alignments import AlignmentEngine


class AlignmentEngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = AlignmentEngine(fs=30, qts=0.35, vas=100)

    def test_reference_values_at_exact_table_row(self):
        self.assertEqual(
            self.engine.get_all_alignments(),
            {
                "QB3": {"vb": 57.5, "fb": 40.5, "f3": 38.1},
                "SBB4": {"vb": 84.7, "fb": 43.8, "f3": 30.0},
                "B4": {"vb": 104.2, "fb": 34.8, "f3": 34.8},
            },
        )

    def test_interpolates_between_table_rows(self):
        result = AlignmentEngine(fs=30, qts=0.375, vas=100).calculate_b4()
        self.assertAlmostEqual(result["vb"], 125.0, places=1)
        self.assertAlmostEqual(result["fb"], 33.5, places=1)
        self.assertAlmostEqual(result["f3"], 33.5, places=1)

    def test_qts_is_clamped_to_supported_table_range(self):
        low = AlignmentEngine(30, 0.10, 100).calculate_qb3()
        edge = AlignmentEngine(30, 0.20, 100).calculate_qb3()
        high = AlignmentEngine(30, 0.90, 100).calculate_qb3()
        upper_edge = AlignmentEngine(30, 0.50, 100).calculate_qb3()
        self.assertEqual(low, edge)
        self.assertEqual(high, upper_edge)


if __name__ == "__main__":
    unittest.main()
