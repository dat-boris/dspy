# Implementation Plan: StructureOfThought Module

**Branch**: `002-structure-of-thought` | **Date**: 2025-12-07 | **Spec**: `specs/002-structure-of-thought/spec.md`
**Input**: Feature specification from `specs/002-structure-of-thought/spec.md`

**Note**: This plan is generated via `/speckit.plan` and describes an incremental implementation path for the `StructureOfThought` module.

## Summary

We will introduce a new `StructureOfThought` program in DSPy that allows users to express a JSON-schema-like structured output directly as part of a signature (similar in spirit to `dspy_so_example.py`), and to obtain strongly structured results from reasoning programs without manual parsing. The core requirement is that `program = dspy.StructureOfThought(ExampleSignature)` behaves like existing DSPy programs while always returning structured outputs that conform to the declared schema, with strict validation and clear error reporting. Implementation will proceed in small, incremental steps: first, add a minimal `StructureOfThought` program that wraps an existing prediction primitive and enforces a schema declared on the signature; next, integrate schema-aware parsing and pretty-printed schema inspection for debugging; then, extend support for optimization workflows (signature updates, extend-only schema compatibility) and add focused examples and tests. Throughout, `StructureOfThought` is designed to plug into existing teleprompting and proposing infrastructure (such as `MIPROv2` and `GroundedProposer`) without requiring any changes to those components, as long as it adheres to the current program/signature conventions (predictors, `get_signature`/`set_signature`, `instructions`, `fields`).

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.10+ (matching existing DSPy baseline)
**Primary Dependencies**: DSPy core (`dspy.Module`, existing prediction/teleprompt primitives), adapter stack (`ChatAdapter`, `JSONAdapter`), and structured validation/models (implemented with Pydantic-style models in accordance with the constitution).
**Storage**: N/A (no persistent storage introduced; reuses existing configuration/history mechanisms).
**Testing**: pytest (extend current `tests/predict` coverage with unit tests and integration-style examples for `StructureOfThought`).
**Target Platform**: Python library usage on typical Linux/macOS environments (same as existing DSPy users).
**Project Type**: Single library project (new module under the existing `dspy` package plus tests and examples).
**Performance Goals**: Overhead of schema validation and pretty-printing is negligible compared to LM calls; `StructureOfThought` should not materially increase per-call latency versus existing `ProgramOfThought` for typical use (validation and parsing should be linear in output size).
**Constraints**: Maintain backwards compatibility with existing prediction programs and adapters; respect DSPy’s adapter abstraction and error-handling patterns; keep the public API small and intuitive (`dspy.StructureOfThought(ExampleSignature)` and result objects with field access).
**Scale/Scope**: Single new program class plus related helpers; 1–2 new source files, a handful of tests and examples, and no new external services.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

Verify compliance with DSPy Constitution principles:

- **Modular Architecture**: `StructureOfThought` will be implemented as a `dspy.Module` (or appropriate program/module subclass), composed from existing prediction primitives and adapters, preserving composability and single-responsibility design.
- **Type Safety**: All new public APIs (constructor, call methods, helpers) will use modern Python type hints, and structured result types will be represented with strongly typed models so that type checking passes across the new module and tests.
- **Adapter Pattern**: LM interactions will continue to flow through the existing adapter stack (`JSONAdapter`/`ChatAdapter`), with `StructureOfThought` focusing on schema declaration and validation rather than direct LM calls.
- **Error Handling**: Parsing and validation failures will raise context-rich custom exceptions (building on existing adapter error types) with clear messages and access to raw/parsed content for debugging; strict mode is the default behavior.
- **Documentation**: The new module and its public surface will include docstrings and at least one dedicated quickstart/example (mirroring `dspy_so_example.py`) that demonstrates defining a schema in the signature and consuming structured outputs.
- **Async Support**: Where existing prediction programs support async (`aforward`/`acall`), `StructureOfThought` will mirror both sync and async interfaces so that structured outputs are available in both modes.
- **Logging**: We will reuse DSPy’s logging and history-tracking mechanisms so that structured outputs and validation errors are visible in history/inspection tools without adding ad-hoc logging patterns.
- **Testing**: New unit tests (for schema definition, validation, and errors) and integration-style tests (for use in teleprompt/optimization loops) will be added, targeting the same level of coverage as existing prediction modules.

**Violations**: None anticipated; if implementation reveals unavoidable deviations, they will be documented in the Complexity Tracking section with justification.

## Project Structure

### Documentation (this feature)

```text
specs/002-structure-of-thought/
├── plan.md              # This file (/speckit.plan output)
├── spec.md              # Feature specification
├── research.md          # Phase 0 output: design decisions and rationale
├── data-model.md        # Phase 1 output: entities and schemas
├── quickstart.md        # Phase 1 output: getting started with StructureOfThought
├── contracts/           # Phase 1 output: API/contracts for the module
└── tasks.md             # Phase 2 output (/speckit.tasks - NOT created by /speckit.plan)
```

### Source Code (repository root)
<!--
  ACTION REQUIRED: Replace the placeholder tree below with the concrete layout
  for this feature. Delete unused options and expand the chosen structure with
  real paths (e.g., apps/admin, packages/something). The delivered plan must
  not include Option labels.
-->

```text
dspy/
├── predict/
│   ├── chain_of_thought.py          # Existing reference implementation
│   ├── program_of_thought.py        # Existing reference implementation
│   └── structure_of_thought.py      # NEW: StructureOfThought program (this feature)
├── adapters/
│   ├── json_adapter.py              # Existing JSON adapter used for structured outputs
│   └── chat_adapter.py              # Existing chat adapter
└── ...

examples/
├── dspy_so_example.py               # Updated/extended example demonstrating StructureOfThought
└── ...

tests/
├── predict/
│   ├── test_program_of_thought.py   # Existing tests
│   └── test_structure_of_thought.py # NEW tests for StructureOfThought
└── ...
```

**Structure Decision**: Extend the existing single-project DSPy layout by adding a new `structure_of_thought.py` module under `dspy/predict/`, updating/adding an example in `examples/dspy_so_example.py`, and creating focused tests in `tests/predict/test_structure_of_thought.py`. No new top-level packages or projects are introduced.

## Complexity Tracking

> **Fill ONLY if Constitution Check has violations that must be justified**

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |
