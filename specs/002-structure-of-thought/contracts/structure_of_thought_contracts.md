# Contracts: StructureOfThought Module

This document captures the intended public API shape for the `StructureOfThought` module at a high level. It is not a final signature but a contract to guide implementation and review.

## High-Level Python API (Conceptual)

### 1. Constructor

- **Signature (conceptual)**:
  - `StructureOfThought(signature: SignatureType, *, strict: bool = True, **config) -> StructureOfThought`
- **Behavior**:
  - Binds the provided `signature` (which includes a structured output schema) to a new program instance.
  - `strict=True` enforces validation errors on invalid outputs.
  - `config` allows future extension (e.g., toggling diagnostics) without breaking the core API.

### 2. Invocation

- **Synchronous**:
  - `__call__(**inputs) -> StructuredResult`
- **Asynchronous**:
  - `acall(**inputs) -> StructuredResult`
- **Behavior**:
  - Accepts input fields as defined by the signature.
  - Uses underlying LM adapters to generate a candidate output.
  - Parses and validates the output against the schema.
  - On success, returns a `StructuredResult` object.
  - On failure, raises a structured validation/parsing error.

### 3. Schema / Result Introspection

- **Schema inspection** (conceptual):
  - `structure_of_thought.schema()` or similar helper → returns a human-readable representation of the active schema (pretty-printed, JSON-like).
- **Example rendering**:
  - `structure_of_thought.example_output()` or a similar utility may be provided to show example structured outputs in the same pretty-printed style.

### 4. Error Contract

- Validation/parsing errors SHOULD:
  - Include the name of the module/program.
  - Include the signature/schema that failed validation.
  - Include the raw LM output (or a safely truncated version) for debugging.
  - Use or extend existing adapter error types where appropriate.

## Notes

- The exact naming of helpers (`schema()`, `example_output()`, etc.) is flexible, but the behaviors above MUST be achievable.
- The constructor and call patterns MUST align with other DSPy prediction programs so that `StructureOfThought` can be dropped into existing workflows with minimal friction.
