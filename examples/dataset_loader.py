"""
Dataset loader utility for DSPy examples.
"""
import json
from pathlib import Path

import dspy


def load_dspy_dataset(
    dataset_name: str,
    split: str = "train",
    input_field: str = "input",
    limit: int = None,
) -> list[dspy.Example]:
    """
    Load a dataset from local JSON file and convert it to DSPy Examples.

    Args:
        dataset_name: Name of the dataset (used to map to local JSON file)
        split: Dataset split to use - "train" takes first 50%, "validate"/"validation" takes last 50%
        input_field: Name of the input field to mark as input
        limit: Optional limit on the number of examples to load (default: None)

    Returns:
        List of dspy.Example objects with the input_field marked as input
    """
    # Hardcoded mapping from dataset_name to JSON file
    dataset_file_map = {
        "tracking_shuffled_objects_five_objects": "bbh_zeroshot_tracking_shuffled_objects_five_objects.json"
    }

    # Get the filename for this dataset
    if dataset_name not in dataset_file_map:
        raise ValueError(
            f"Dataset '{dataset_name}' not found in mapping. "
            f"Available datasets: {list(dataset_file_map.keys())}"
        )

    filename = dataset_file_map[dataset_name]

    # Get the path to the examples directory (where this file is located)
    current_dir = Path(__file__).parent
    json_path = current_dir / filename

    if not json_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {json_path}")

    # Load JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Convert to dspy.Example objects
    examples = []
    for item in data:
        # Create Example with all fields from the JSON item
        # Match the pattern used in DataLoader: create from dict then mark inputs
        example = dspy.Example(item).with_inputs(input_field)
        examples.append(example)

    # Split the data based on split parameter
    total_examples = len(examples)
    split_point = total_examples // 2

    if split == "train":
        # Take first 50%
        examples = examples[:split_point]
    elif split in ("validate", "validation"):
        # Take last 50% (support both "validate" and "validation" for compatibility)
        examples = examples[split_point:]
    else:
        raise ValueError(
            f"Invalid split '{split}'. Must be 'train' or 'validate'/'validation'."
        )

    # Apply limit if specified
    if limit is not None and limit > 0:
        examples = examples[:limit]

    return examples
