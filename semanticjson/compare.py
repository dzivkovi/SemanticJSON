"""
Script designed for semantically comparing JSON objects. It goes beyond traditional diffing tools
by incorporating deep learning techniques to understand the meaning of text within JSON data.
"""
import json
import argparse
from deepdiff import DeepDiff
from sentence_transformers import SentenceTransformer, util
from colorama import Fore, Style
from tabulate import tabulate


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


def color_print_diffs(differences):
    """
    Pretty print differences with color highlights.
    """
    # Structural differences
    print(Fore.CYAN + "Structural Differences:" + Style.RESET_ALL)
    if not differences["structural_diff"]:
        print(Fore.GREEN + "  None" + Style.RESET_ALL)
    else:
        print(json.dumps(differences["structural_diff"], indent=2))

    # Semantic differences
    print(Fore.CYAN + "\nSemantic Differences (Similarity Scores):" + Style.RESET_ALL)
    if not differences["semantic_diff"]:
        print(Fore.GREEN + "  None" + Style.RESET_ALL)
    else:
        for key, val in differences["semantic_diff"].items():
            # Highlight in red if similarity < 0.8
            if val < 0.8:
                print(Fore.RED + f"  {key}: {val}" + Style.RESET_ALL)
            else:
                print(Fore.YELLOW + f"  {key}: {val}" + Style.RESET_ALL)


def table_print_diffs(differences):
    """
    Print differences in tabular form for better readability.
    """
    table_data = []
    # Structural diffs
    if differences["structural_diff"]:
        table_data.append(["Structural Differences", "---", "---"])
        # Convert structural_diff dict to a readable table
        for diff_type, details in differences["structural_diff"].items():
            table_data.append([diff_type, str(details), "Structural difference"])
    else:
        table_data.append(["Structural Differences", "None", "No difference"])

    # Semantic diffs
    if differences["semantic_diff"]:
        table_data.append(["Semantic Differences", "---", "---"])
        for key, score in differences["semantic_diff"].items():
            explanation = "Low similarity" if score < 0.8 else "High similarity"
            table_data.append([key, f"{score:.2f}", explanation])
    else:
        table_data.append(["Semantic Differences", "None", "No difference"])

    print(tabulate(table_data, headers=["Key/Section", "Value/Score", "Explanation"], tablefmt="fancy_grid"))


def main():
    """
    Compare two JSON files using DeepDiff and sentence-transformers.
    """
    parser = argparse.ArgumentParser(
        description="Compare two JSON files using DeepDiff and sentence-transformers."
    )
    parser.add_argument("json1", help="Path to the first JSON file.")
    parser.add_argument("json2", help="Path to the second JSON file.")
    parser.add_argument(
        "--format",
        choices=["color", "colour", "table", "raw"],
        default="raw",
        help="Output format: color, table, or raw."
    )
    args = parser.parse_args()

    results = hybrid_json_compare(args.json1, args.json2)

    if args.format == "color" or args.format == "colour":
        color_print_diffs(results)
    elif args.format == "table":
        table_print_diffs(results)
    else:
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
