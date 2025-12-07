# Quickstart: Using StructureOfThought

This guide shows how to define a simple signature with a JSON-like structured output, wrap it in `StructureOfThought`, and consume the result.

## 1. Define a signature with structured output and reasoning JSON

At a high level, you define a signature that describes the input fields, the desired structured output fields (for example, `answer` and `confidence`), and a richer `reasoning_json` whose shape is described by a `reasoning_json_schema`. The exact syntax will mirror existing DSPy signature patterns but ensure both the output side and the reasoning JSON are treated as schemas, not just free-form text.

Conceptually:

- Inputs: problem description, context, or question.
- Outputs (schema):
  - `answer: str`
  - `confidence: float | None`
- Reasoning JSON (schema):
  - `reasoning_json_schema` that describes a structured `reasoning_json` object, which includes at least the primary output fields and may add extra detail used for debugging, optimization, or downstream tools.

## 2. Construct a StructureOfThought program

You then construct a program using the signature:

- Create a signature class (or equivalent) that includes the output schema.
- Instantiate the program: `program = dspy.StructureOfThought(ExampleSignature)`.

The program behaves like other DSPy programs but always returns a structured result object that matches the schema.

## 3. Call the program and consume structured results

When you call the program with inputs:

- The underlying LM and adapters generate a candidate output.
- `StructureOfThought` parses and validates the primary output against the output schema and validates the `reasoning_json` against the `reasoning_json_schema`.
- On success, you receive a structured result object with predictable fields (e.g., `result.answer`, `result.confidence`, `result.reasoning_json`).
- On failure, a clear, structured error is raised rather than returning a partial or inconsistent object.

## 4. Debugging with pretty-printed schema and outputs

For debugging and documentation, `StructureOfThought` will provide ways to:

- Inspect the active schema in a human-readable, indented format (JSON-like).
- View example structured outputs in the same pretty-printed style.

This makes it easy to reason about what the program is expected to produce and how real outputs compare.

## 5. Iterating on the schema

As you refine your application or run teleprompting/optimization workflows:

- You can update the signature/schema to add new fields or refine structure.
- `StructureOfThought` enforces an extend-only compatibility rule: new schemas may add fields but should not remove or weaken existing ones.
- Downstream tools can continue to rely on previously declared fields remaining available and correctly typed.

For a concrete example, the plan is to extend `dspy_so_example.py` to demonstrate `StructureOfThought` end-to-end using a real signature and a small dataset.
