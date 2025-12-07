#! /usr/bin/env python3
"""
DSPy Natural Language Example
"""
import re

from pydantic import BaseModel, Field
import dspy

from examples.dataset_loader import load_dspy_dataset


# Points at LiteLLM proxy
dspy.configure(
    lm=dspy.LM("openai/claude-haiku-3", api_base="http://0.0.0.0:4000"),
    adapter=dspy.JSONAdapter()  # Use JSONAdapter for structured output
)

# Avoid disk cache to cause churn in rerunning the script
dspy.configure_cache(enable_disk_cache=False)

# NOTE: we need to use a smarter model for optimization since the base model could be not good at reasoning
OPTIMIZATION_MODEL = dspy.LM("openai/gpt-5.1", api_base="http://0.0.0.0:4000")


class StructoResponse(BaseModel):
    """Structured response container with reasoning and answer fields."""
    # NOTE: we intentionally put answer first so we can ensure that the order
    # is optimized when the program is used.
    answer: str = Field(description="The final answer to the question - give the answer in format of (A), (B), (C), (D), or (E)")
    reasoning: str = Field(description="The reasoning process for the answer")


class ExampleSignature(dspy.Signature):
    """Track the shuffled input and give reason. On the final sentence, return the answer."""

    input: str = dspy.InputField()
    response: StructoResponse = dspy.OutputField(desc="Structured response with reasoning and answer")

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
    matches = re.findall(pattern, pred.response.answer)
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
    program = dspy.StructureOfThought(ExampleSignature)
    answer = program(input=test_input)
    print("=== StructureOfThought Schema ===")
    print(program.schema_repr())
    print("=== Single call StructuredResult ===")
    print(answer.pretty())
    print(f"Single call answer: {answer.response.answer}")

    # Extract answer using regex
    pattern = r"\(([A-Z])\)"
    matches = re.findall(pattern, answer.response.answer)
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
):
    """
    Optimize a DSPy program using MIPROv2 with minimal cycles.

    Args:
        dataset_name: Name of the dataset to load
        input_field: Name of the input field
        num_threads: Number of threads to use
        auto: Auto mode for MIPROv2 (default: "light" for minimal cycles)
        limit: Optional limit on the number of examples to load (default: None)

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
    base_program = dspy.StructureOfThought(ExampleSignature)

    # Create a metric function
    def metric_with_filter(example, pred, trace=None):
        return validate_answer(example, pred, trace)

    # Optimize the program with MIPROv2 - minimal cycles
    teleprompter = dspy.MIPROv2(
        metric=metric_with_filter,
        auto=auto,
        num_threads=num_threads,
        prompt_model=OPTIMIZATION_MODEL,
    )
    optimized_program = teleprompter.compile(base_program, trainset=trainset)
    print(f"Optimized program: {optimized_program}")

    return optimized_program


def refine_schema_example():
    """Demonstrate a simple extend-only schema refinement for StructureOfThought."""

    class ExtendedStructoResponse(StructoResponse):
        confidence: float | None = Field(
            default=None,
            description="Optional confidence score for the selected answer.",
        )

    class ExtendedSignature(dspy.Signature):
        """Extended signature that adds a confidence field to the structured response."""

        input: str = dspy.InputField()
        response: ExtendedStructoResponse = dspy.OutputField(
            desc="Structured response with reasoning, answer, and confidence",
        )

    program = dspy.StructureOfThought(ExampleSignature)
    print("=== Original StructureOfThought Schema ===")
    print(program.schema_repr())

    # Rebind to an extend-only compatible schema (ExtendedStructoResponse subclass).
    program.rebind_signature(ExtendedSignature)
    print("=== Refined StructureOfThought Schema (ExtendedSignature) ===")
    print(program.schema_repr())


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
    # Example usage
    dataset_name = "tracking_shuffled_objects_five_objects"
    test_input = "Alice, Bob, Claire, Dave, and Eve are dancers at a square dance. At the start of a song, they each have a partner: Alice is dancing with Ophelia, Bob is dancing with Jamie, Claire is dancing with Melissa, Dave is dancing with Rodrigo, and Eve is dancing with Patrick.\nThroughout the song, the dancers often trade partners. First, Claire and Bob switch partners. Then, Claire and Eve switch partners. Then, Claire and Bob switch partners. Then, Eve and Dave switch partners. Finally, Claire and Alice switch partners. At the end of the dance, Alice is dancing with\nOptions:\n(A) Ophelia\n(B) Jamie\n(C) Melissa\n(D) Rodrigo\n(E) Patrick"

    # 1. Single evaluation example
    print("=" * 60)
    print("1. Single Evaluation Example")
    print("=" * 60)
    base_program = single_eval_example(test_input, dataset_name=dataset_name)

    # Evaluate the base program
    # print("\n" + "=" * 60)
    # print("1. Evaluating Base Program (20 examples)")
    # print("=" * 60)
    # accuracy = evaluate(base_program, limit=20)
    # print(f"Base program accuracy: {accuracy}")

    # 2. Optimize the program
    print("\n" + "=" * 60)
    print("2. Optimizing Program (MIPROv2 - minimal cycles, 20 examples)")
    print("=" * 60)
    optimized_program = optimize(auto="light", num_threads=1, limit=20)

    # Test optimized program
    answer = optimized_program(input=test_input)
    print(f"Optimized program answer: {answer}")

    # 3. Evaluate the optimized program
    print("\n" + "=" * 60)
    print("3. Evaluating Program (20 examples)")
    print("=" * 60)
    accuracy = evaluate(optimized_program, limit=20)
    print(f"Final accuracy: {accuracy}")
