"""
tests/test_compare.py
"""
import unittest
import os
from semanticjson.compare import hybrid_json_compare


class TestHybridJsonCompare(unittest.TestCase):
    """
    Unit tests for the hybrid_json_compare function in semanticjson/compare.py.
    """
    def test_compare_files(self):
        """
        Unit test comparing two known JSON files (file1.json and file2.json)
        and asserting the expected structural and semantic differences.
        """
        file1 = os.path.join("tests", "test_data", "file1.json")
        file2 = os.path.join("tests", "test_data", "file2.json")

        results = hybrid_json_compare(file1, file2)
        self.assertIn("structural_diff", results, "Results should contain 'structural_diff'.")
        self.assertIn("semantic_diff", results, "Results should contain 'semantic_diff'.")

        structural_diff = results["structural_diff"]
        self.assertIn("values_changed", structural_diff, "Structural diff should contain 'values_changed'.")

        # Verify that a specific path changed
        changed_values = structural_diff["values_changed"]
        self.assertIn("root['input_data'][1]['business_name']", changed_values, "Expected path not found in values_changed.")

        # Check old_value and new_value for one of the changed entries
        changed_item = changed_values["root['input_data'][1]['business_name']"]
        self.assertEqual(changed_item["old_value"], "", "Old value should be empty string.")
        self.assertEqual(changed_item["new_value"], "LORETTA Inc", "New value should be 'LORETTA Inc'.")

        # Check that no semantic differences were detected
        self.assertEqual(results["semantic_diff"], {}, "Expected no semantic differences for these test files.")


if __name__ == "__main__":
    unittest.main()
