#! /usr/bin/env python3
"""
DSPy Natural Language Example
"""
import argparse
import re

import dspy
from dspy.teleprompt import COPRO

from examples.dataset_loader import load_dspy_dataset


EXPERIMENT_NAME = "DSPy-BaselineNL"
EXPERIMENT_MODEL = "claude-haiku-3"
DATASET_NAME = "shuffle_five"


# Enable MLflow autologging for DSPy optimization tracking (optional)
try:
    import mlflow

    mlflow.dspy.autolog(
        log_compiles=True,    # Track optimization process
        log_evals=True,       # Track evaluation results
        log_traces_from_compile=True  # Track program traces during optimization
    )

    # Configure MLflow tracking
    mlflow.set_tracking_uri("http://localhost:5000")  # Use local MLflow server
    mlflow.set_experiment(f"{EXPERIMENT_NAME}-{DATASET_NAME}-{EXPERIMENT_MODEL}")
except ImportError:
    print("MLflow not available, continuing without tracking")
    pass


# Points at LiteLLM proxy
dspy.configure(
    lm=dspy.LM(f"openai/{EXPERIMENT_MODEL}", api_base="http://0.0.0.0:4000"),
)

# NOTE: we need to use a smarter model for optimization since the base model could be not good at reasoning
OPTIMIZATION_MODEL = dspy.LM("openai/gpt-5.1", api_base="http://0.0.0.0:4000")

# Avoid disk cache to cause churn in rerunning the script
dspy.configure_cache(enable_disk_cache=False)


class ExampleSignature(dspy.Signature):
    """Solve the program task."""

    input: str = dspy.InputField()
    answer: str = dspy.OutputField()


def validate_answer(example, pred, trace=None):
    """
    Compare the target field from the example with the answer from the prediction.

    Args:
        example: The example with the target answer
        pred: The prediction from the model
        trace: Optional trace (unused)
    """
    target = getattr(example, "target", "")

    # Extract answer from prediction using regex to find last instance of (A), (B), etc.
    pattern = r"\(([A-Z])\)"
    matches = re.findall(pattern, pred.answer)
    if not matches:
        return False

    # Get the last match (e.g., "A" from "(A)")
    pred_answer = f"({matches[-1]})"

    return target.lower() == pred_answer.lower()


def single_eval_example(test_input: str, dataset_name: str = "tracking_shuffled_objects_five_objects"):
    """
    Run a single evaluation example with the base program.

    Args:
        test_input: Input string to test
        dataset_name: Name of the dataset (unused, kept for API compatibility)

    Returns:
        The prediction result
    """
    program = dspy.ChainOfThought(ExampleSignature)
    answer = program(input=test_input)
    print(f"Single call answer: {answer.answer}")

    # Extract answer using regex
    pattern = r"\(([A-Z])\)"
    matches = re.findall(pattern, answer.answer)
    if matches:
        extracted = f"({matches[-1]})"
        print(f"Single call answer (extracted): {extracted}")
    else:
        print("Single call answer (extracted): No match found")

    return program


def optimize(
    dataset_name: str = "tracking_shuffled_objects_five_objects",
    input_field: str = "input",
    num_threads: int = 1,
    auto: str = "light",
    limit: int = None,
    use_copro: bool = False,
):
    """
    Optimize a DSPy program using MIPROv2 with minimal cycles.

    Args:
        dataset_name: Name of the dataset to load
        split: Dataset split to use
        input_field: Name of the input field
        num_threads: Number of threads to use
        auto: Auto mode for MIPROv2 (default: "light" for minimal cycles)
        limit: Optional limit on the number of examples to load (default: None)
        use_copro: If True, use COPRO optimizer instead of MIPROv2

    Returns:
        Optimized DSPy program
    """
    # Load training dataset
    trainset = load_dspy_dataset(
        dataset_name=dataset_name,
        split="train",
        input_field=input_field,
        limit=limit
    )

    # Create base program
    base_program = dspy.ChainOfThought(ExampleSignature)

    # Create a metric function
    def metric_with_filter(example, pred, trace=None):
        return validate_answer(example, pred, trace)

    # Optimize the program with the selected optimizer (MIPROv2 by default, or COPRO if requested)
    if use_copro:
        teleprompter = COPRO(
            metric=metric_with_filter,
            prompt_model=OPTIMIZATION_MODEL,
            breadth=8,  # Set to 8 to ensure n <= 8 (COPRO uses n=breadth or n=breadth-1)
            init_temperature=1.0,  # Set to 1.0 instead of default 1.4
        )
        optimized_program = teleprompter.compile(
            base_program,
            trainset=trainset,
            eval_kwargs={
                "num_threads": num_threads,
                "display_progress": True,
                "display_table": 0,
            },
        )
    else:
        teleprompter = dspy.MIPROv2(
            metric=metric_with_filter,
            auto=auto,
            num_threads=num_threads,
            prompt_model=OPTIMIZATION_MODEL,
        )
        optimized_program = teleprompter.compile(base_program, trainset=trainset)
    print(f"Optimized program: {optimized_program}")

    return optimized_program


def evaluate(
    program,
    dataset_name: str = "tracking_shuffled_objects_five_objects",
    input_field: str = "input",
    num_threads: int = 1,
    display_progress: bool = True,
    display_table: int = 5,
    provide_traceback: bool = True,
    limit: int = None,
):
    """
    Evaluate a DSPy program on a dataset.

    Args:
        program: The DSPy program to evaluate
        dataset_name: Name of the dataset to load
        split: Dataset split to use
        input_field: Name of the input field
        num_threads: Number of threads to use for evaluation
        display_progress: Whether to display progress
        display_table: Number of examples to display in the table
        provide_traceback: Whether to provide traceback for debugging (default: True)
        limit: Optional limit on the number of examples to load (default: None)

    Returns:
        Accuracy score
    """
    # Load evaluation dataset
    eval_set = load_dspy_dataset(
        dataset_name=dataset_name,
        split="validate",
        input_field=input_field,
        limit=limit
    )

    # Create a metric function
    def metric_with_filter(example, pred, trace=None):
        return validate_answer(example, pred, trace)

    # Evaluate the program
    evaluator = dspy.Evaluate(
        devset=eval_set,
        num_threads=num_threads,
        display_progress=display_progress,
        display_table=display_table,
        provide_traceback=provide_traceback,
    )
    accuracy = evaluator(program, metric=metric_with_filter)
    print(f"Program accuracy: {accuracy}")

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DSPy Baseline NL example")
    parser.add_argument(
        "--copro",
        action="store_true",
        help="Use COPRO optimizer instead of MIPROv2 for optimization.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Limit on the number of examples to use for optimization and evaluation (default: 20).",
    )
    args = parser.parse_args()

    # Example usage
    dataset_name = "tracking_shuffled_objects_five_objects"
    test_input = "Alice, Bob, Claire, Dave, and Eve are dancers at a square dance. At the start of a song, they each have a partner: Alice is dancing with Ophelia, Bob is dancing with Jamie, Claire is dancing with Melissa, Dave is dancing with Rodrigo, and Eve is dancing with Patrick.\nThroughout the song, the dancers often trade partners. First, Claire and Bob switch partners. Then, Claire and Eve switch partners. Then, Claire and Bob switch partners. Then, Eve and Dave switch partners. Finally, Claire and Alice switch partners. At the end of the dance, Alice is dancing with\nOptions:\n(A) Ophelia\n(B) Jamie\n(C) Melissa\n(D) Rodrigo\n(E) Patrick"

    # 1. Single evaluation example
    print("=" * 60)
    print("1. Single Evaluation Example")
    print("=" * 60)
    base_program = single_eval_example(test_input, dataset_name=dataset_name)

    # 2. Optimize the program
    print("\n" + "=" * 60)
    print(f"2. Optimizing Program (MIPROv2 or COPRO - minimal cycles, {args.limit} examples)")
    print("=" * 60)
    optimized_program = optimize(
        auto="light",
        num_threads=1,
        limit=args.limit,
        use_copro=args.copro,
    )

    # Test optimized program
    answer = optimized_program(input=test_input)
    print(f"Optimized program answer: {answer}")

    # 3. Evaluate the optimized program
    print("\n" + "=" * 60)
    print(f"3. Evaluating Program ({args.limit} examples)")
    print("=" * 60)
    accuracy = evaluate(optimized_program, limit=args.limit)
    print(f"Final accuracy: {accuracy}")
