# SemanticJSON

**SemanticJSON** is a powerful Python library designed for **semantically comparing JSON objects**. It goes beyond traditional diffing tools by incorporating deep learning techniques to understand the meaning of text within JSON data.

## Key Features

* **Hybrid Comparison:** Combines traditional syntactic diffing with semantic analysis using Sentence Transformers.
* **Deep Semantic Understanding:** Leverages Large Language Models (LLMs) to generate embeddings and compare the meaning of text values, even if the wording is different.
* **Layered Results:** Presents results in a hierarchical manner, allowing users to first see structural differences and then "zoom in" to specific text values to understand their semantic similarity.
* **Reduced False Positives:** Minimizes instances where tools flag changes as significant when they are merely paraphrases or semantically equivalent expressions.
* **Improved Accuracy:** Enables more accurate identification of truly meaningful changes, which is crucial for tasks like version control, data analysis, and change tracking.
* **Enhanced User Experience:** Offers a layered presentation of results, allowing users to focus on the most relevant differences and explore semantic similarities as needed.

## Value Proposition

SemanticJSON offers a more nuanced and accurate way to compare JSON objects, addressing the limitations of traditional diffing tools that focus solely on syntactic differences. This is particularly valuable in scenarios where:

* **Textual data is crucial:**  When the meaning of text within JSON objects is important for understanding the differences.
* **False positives are problematic:** When it's critical to avoid flagging minor wording changes as significant differences.
* **Accuracy is paramount:** When precise identification of meaningful changes is essential.

## Getting Started

```bash
pip install sentence-transformers deepdiff
```

```python
import json
from deepdiff import DeepDiff
from sentence_transformers import SentenceTransformer, util

def hybrid_json_compare(json1_path, json2_path):
    """
    Compares two JSON files using a hybrid approach.

    Args:
        json1_path (str): Path to the first JSON file.
        json2_path (str): Path to the second JSON file.

    Returns:
        dict: A dictionary containing both structural and semantic differences.
    """

    with open(json1_path, 'r') as f1, open(json2_path, 'r') as f2:
        json1 = json.load(f1)
        json2 = json.load(f2)

    # Lexical comparison using DeepDiff
    structural_diff = DeepDiff(json1, json2)

    # Semantic comparison using Sentence Transformers
    model = SentenceTransformer('all-MiniLM-L6-v2')
    semantic_diff = {}
    for key in json1:
        if isinstance(json1, str) and isinstance(json2, str):
            embedding1 = model.encode(json1, convert_to_tensor=True)
            embedding2 = model.encode(json2, convert_to_tensor=True)
            similarity = util.pytorch_cos_sim(embedding1, embedding2)
            semantic_diff = similarity.item()

    return {
        "structural_diff": structural_diff,
        "semantic_diff": semantic_diff
    }

# Example usage
results = hybrid_json_compare('file1.json', 'file2.json')
print(results)
```

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License.
