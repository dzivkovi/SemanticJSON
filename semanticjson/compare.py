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


def hybrid_json_compare(json1_path, json2_path, threshold=0.9):
    """
    Compares two JSON files using a hybrid approach: structural (DeepDiff)
    and a semantic check (sentence-transformers).

    1. Uses DeepDiff to identify which fields changed (structural_diff).
    2. For each changed field that is a string, calculates semantic similarity.
    3. If similarity >= threshold, removes that entry from structural_diff
       and marks it as "Equivalent" in semantic_diff.
    4. Otherwise, flags it as "Changed" in semantic_diff.
    """

    with open(json1_path, "r", encoding="utf-8") as f1, open(json2_path, "r", encoding="utf-8") as f2:
        json1 = json.load(f1)
        json2 = json.load(f2)

    # Structural comparison
    structural_diff = DeepDiff(json1, json2)

    # Initialize sentence-transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Dictionary to store our semantic analysis
    # Key: path (e.g., "root['my_field']"), Value: dict with "similarity", "status", etc.
    semantic_diff = {}

    # If there are changes in values_changed, evaluate them semantically
    if "values_changed" in structural_diff:
        for path, changes in list(structural_diff["values_changed"].items()):
            old_val = changes.get("old_value")
            new_val = changes.get("new_value")

            # Only if both old and new values are strings
            if isinstance(old_val, str) and isinstance(new_val, str):
                embedding1 = model.encode(old_val, convert_to_tensor=True)
                embedding2 = model.encode(new_val, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(embedding1, embedding2).item()

                # If above threshold, remove from structural_diff and note as equivalent
                if similarity >= threshold:
                    del structural_diff["values_changed"][path]
                    semantic_diff[path] = {
                        "status": "Equivalent (semantically)",
                        "similarity": similarity,
                        "old_value": old_val,
                        "new_value": new_val,
                    }
                else:
                    semantic_diff[path] = {
                        "status": "Changed (semantically different)",
                        "similarity": similarity,
                        "old_value": old_val,
                        "new_value": new_val,
                    }

        # If "values_changed" became empty, remove it from structural_diff
        if not structural_diff["values_changed"]:
            del structural_diff["values_changed"]

    return {
        "structural_diff": structural_diff,
        "semantic_diff": semantic_diff
    }


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
    print(Fore.CYAN + "\nSemantic Differences:" + Style.RESET_ALL)
    if not differences["semantic_diff"]:
        print(Fore.GREEN + "  None" + Style.RESET_ALL)
    else:
        for path, info in differences["semantic_diff"].items():
            similarity = info["similarity"]
            old_value = info["old_value"]
            new_value = info["new_value"]
            status = info["status"]

            if status == "Equivalent (semantically)":
                # High similarity => print in yellow
                print(Fore.YELLOW + f"  {path} => {similarity:.2f} [Equivalent]" + Style.RESET_ALL)
            else:
                # Low similarity => print in red
                print(Fore.RED + f"  {path} => {similarity:.2f} [Changed]" + Style.RESET_ALL)
            print("    Old:", old_value)
            print("    New:", new_value)


def table_print_diffs(differences):
    """
    Print a simplified, Excel-style table view with columns:
    Path, Old Value, New Value, Similarity, and Status.

    This version skips printing a separate structural row
    for paths that also have semantic differences.
    """
    structural_diff = differences.get("structural_diff", {})
    semantic_diff = differences.get("semantic_diff", {})

    table_data = [
        ["Path", "Old Value", "New Value", "Similarity", "Status"]
    ]

    # 1. Print structural diffs only if they are not in the semantic diff
    if "values_changed" in structural_diff:
        for path, change_info in structural_diff["values_changed"].items():
            if path in semantic_diff:
                # Skip, because we'll show it in semantic section
                continue
            old_val = change_info.get("old_value", "")
            new_val = change_info.get("new_value", "")
            table_data.append([path, old_val, new_val, "-", "Structural difference"])

    # 2. Print semantic diffs
    #    (includes both "Equivalent" and "Changed" entries)
    for path, info in semantic_diff.items():
        old_val = info.get("old_value", "")
        new_val = info.get("new_value", "")
        similarity = f"{info['similarity']:.2f}"
        status = info["status"]
        table_data.append([path, old_val, new_val, similarity, status])

    # 3. If no rows beyond header, print a "no differences" entry
    if len(table_data) == 1:
        table_data.append(["None", "-", "-", "-", "No differences found"])

    print(tabulate(table_data, headers="firstrow", tablefmt="simple"))


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
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help="Similarity threshold above which differences are considered 'semantically equivalent'."
    )
    args = parser.parse_args()

    results = hybrid_json_compare(args.json1, args.json2, threshold=args.threshold)

    if args.format == "color" or args.format == "colour":
        color_print_diffs(results)
    elif args.format == "table":
        table_print_diffs(results)
    else:
        # Raw JSON output
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
