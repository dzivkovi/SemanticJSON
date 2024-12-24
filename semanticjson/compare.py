"""
Script designed for semantically comparing JSON objects. It goes beyond traditional diffing tools
by incorporating deep learning techniques to understand the meaning of text within JSON data.
"""
import json
import argparse
from deepdiff import DeepDiff
from sentence_transformers import SentenceTransformer, util


def hybrid_json_compare(json1_path, json2_path):
    """
    Compares two JSON files using a hybrid approach: structural (DeepDiff)
    and a simple semantic check (sentence-transformers).
    """

    with open(json1_path, "r", encoding="utf-8") as f1, open(json2_path, "r", encoding="utf-8") as f2:
        json1 = json.load(f1)
        json2 = json.load(f2)

    # Structural comparison
    structural_diff = DeepDiff(json1, json2)

    # Simple semantic comparison: checks if top-level values are strings and compares them
    model = SentenceTransformer('all-MiniLM-L6-v2')
    semantic_diff = {}

    # Only applies if json1 and json2 are dicts with string values at top level
    if isinstance(json1, dict) and isinstance(json2, dict):
        for key in set(json1.keys()).intersection(json2.keys()):
            val1 = json1[key]
            val2 = json2[key]
            if isinstance(val1, str) and isinstance(val2, str):
                embedding1 = model.encode(val1, convert_to_tensor=True)
                embedding2 = model.encode(val2, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(embedding1, embedding2).item()
                semantic_diff[key] = similarity

    return {"structural_diff": structural_diff, "semantic_diff": semantic_diff}


def main():
    """
    Compare two JSON files using DeepDiff and sentence-transformers.
    """
    parser = argparse.ArgumentParser(description="Compare two JSON files using DeepDiff and sentence-transformers.")
    parser.add_argument("json1", help="Path to the first JSON file.")
    parser.add_argument("json2", help="Path to the second JSON file.")
    args = parser.parse_args()

    results = hybrid_json_compare(args.json1, args.json2)
    print(results)


if __name__ == "__main__":
    main()
