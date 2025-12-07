# Tasks: StructureOfThought Module

**Input**: Design documents from `specs/002-structure-of-thought/`
**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`

**Organization**: Tasks are grouped by user story to enable independent, incremental implementation.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish minimal scaffolding and file layout for `StructureOfThought`.

- [x] T001 [P] [Shared] Create `dspy/predict/structure_of_thought.py` with a placeholder `StructureOfThought` class and module-level logger.
- [x] T002 [P] [Shared] Ensure `StructureOfThought` is importable from the `dspy` package (e.g., update `dspy/__init__.py` or equivalent export location).
- [x] T003 [P] [Shared] Add a skeleton test file `tests/predict/test_structure_of_thought.py` with a basic import-and-instantiation smoke test.

**Checkpoint**: Codebase compiles/imports with a stub `StructureOfThought` and basic test scaffold in place.

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core data structures, schema logic, structured prediction primitive, schema-driven adapter behavior, and documentation of how they integrate with teleprompters/proposers.

**⚠️ CRITICAL**: No user story work should begin until this phase is complete.

- [x] T010 [US-Foundation] Define internal data structures for `OutputSchema` and `reasoning_json_schema` in `dspy/predict/structure_of_thought.py`, aligned with `data-model.md` (primary outputs + reasoning JSON superset rule).
- [x] T011 [US-Foundation] Implement a `StructuredResult` container type in `dspy/predict/structure_of_thought.py` with fields for primary outputs, `reasoning_json`, and validation metadata.
- [x] T012 [P] [US-Foundation] Wire validation to a StructureOfThought-specific adapter (likely built on top of `dspy.JSONAdapter` or similar JSON-oriented tooling) without yet exposing a full user-facing API.
- [x] T013 [P] [US-Foundation] Add unit tests in `tests/predict/test_structure_of_thought.py` that validate basic schema enforcement and strict error behavior on clearly invalid outputs.
- [x] T014 [US-Foundation] Introduce a new `PredictWithStructure` primitive in `dspy/predict/structure_of_thought.py` as a subclass of `dspy.Predict` that: (a) enforces Pydantic output models; (b) computes a JSON schema for the output structure (including any `reasoning_json`); and (c) sets its own `Signature.instructions` string to that pretty-printed JSON schema.
- [x] T015 [P] [US-Foundation] Refactor `StructureOfThought` to stop constructing plain `dspy.Predict` instances and instead compose one or more `PredictWithStructure` predictors from the provided signature, so that teleprompters/proposers still see standard predictors but their `instructions` contain the output JSON schema.
- [x] T016 [P] [US-Foundation] Ensure `StructureOfThought.rebind_signature` delegates schema evolution to the underlying `PredictWithStructure` predictors (updating their Signatures and `instructions`), and enforces extend-only compatibility when updating the output schema.
- [x] T017 [P] [US-Foundation] Add tests in `tests/predict/test_structure_of_thought.py` that verify, for a `StructureOfThought` built on `PredictWithStructure`, that `get_signature(program.predictors()[i]).instructions` is valid JSON matching the output schema (before and after `rebind_signature`).
- [x] T018 [P] [US-Foundation] Update the class-level docstrings and key inline comments in `dspy/predict/structure_of_thought.py` (for both `PredictWithStructure` and `StructureOfThought`) to clearly document: (a) that `PredictWithStructure` sets `Signature.instructions` to the output JSON schema, and (b) that teleprompters/proposers (e.g., `GroundedProposer`) are expected to read this schema via `get_signature(predictor).instructions` when constructing `basic_instruction`.
- [x] T019 [P] [US-Foundation] Design and implement a schema-driven adapter in `dspy/adapters/` (e.g., `StructureJSONAdapter` subclassing `JSONAdapter`) that, when used, builds its Pydantic response model from a JSON schema string (e.g., `signature.instructions`) instead of inferring solely from `signature.output_fields`.
- [x] T020 [P] [US-Foundation] Integrate the schema-driven adapter with `PredictWithStructure` (and thus `StructureOfThought`) by ensuring that, when these modules call the LM, they do so inside a `settings.context(adapter=StructureJSONAdapter(...))` or via configuration, so that structured outputs are validated strictly against the JSON schema.
- [x] T021 [P] [US-Foundation] Add tests in `tests/predict/test_structure_of_thought.py` that exercise `PredictWithStructure`/`StructureOfThought` together with the schema-driven adapter: (a) LM responses that match the JSON schema are accepted and parsed correctly; (b) LM responses that violate the schema (missing required fields, wrong types, extra fields) raise clear parse/validation errors.

**Checkpoint**: The structured prediction primitive (`PredictWithStructure`), schema logic, and strict validation exist and are test-covered; `StructureOfThought` composes them and is ready for user stories.

---

## Phase 3: User Story 1 - Define a structured reasoning program (Priority: P1) 🎯 MVP

**Goal**: Allow users to define a `StructureOfThought` program via `dspy.StructureOfThought(ExampleSignature)` that returns structured outputs plus `reasoning_json` validated against `reasoning_json_schema`.

**Independent Test**: A simple signature and program can be defined, invoked on a toy input, and produce a `StructuredResult` with `answer`, `confidence`, and `reasoning_json` fields populated and validated.

### Implementation for User Story 1

- [x] T020 [US1] Implement the public `StructureOfThought` constructor in `dspy/predict/structure_of_thought.py` that accepts a signature with both output schema and `reasoning_json_schema`.
- [x] T021 [US1] Implement the synchronous call path (`__call__` / `forward`) to run the underlying LM/program, parse outputs into `StructuredResult`, and enforce strict validation (error on mismatch).
- [x] T022 [P] [US1] Add pretty-printed schema and example-output inspection helpers (e.g., `schema_repr` / similar) for debugging, using indented JSON-like formatting.
- [x] T023 [P] [US1] Extend `tests/predict/test_structure_of_thought.py` with end-to-end tests that instantiate a toy signature, run `StructureOfThought`, and assert on `StructuredResult` fields and error paths.
- [x] T024 [P] [US1] Add or update a minimal example path in `examples/dspy_so_example.py` to demonstrate constructing and calling a `StructureOfThought` program with a simple schema.

**Checkpoint**: User Story 1 is implementable and testable independently; users can get structured outputs plus validated `reasoning_json` from a simple example.

---

## Phase 4: User Story 2 - Evolve schemas during optimization (Priority: P2)

**Goal**: Support workflows (e.g., MIPRO-style optimizers) that update the signature and `reasoning_json_schema` over time while keeping the public API stable and enforcing extend-only compatibility.

**Independent Test**: An optimization-like loop can update the signature/schema between iterations, and each iteration’s `StructureOfThought` program still returns outputs and `reasoning_json` matching the latest schema, without breaking existing fields.

### Implementation for User Story 2

- [x] T030 [US2] Add mechanisms in `dspy/predict/structure_of_thought.py` to bind and rebind updated signatures/`reasoning_json_schema` (e.g., via helper methods or integration points used by teleprompt/optimizers).
- [x] T031 [US2] Implement extend-only compatibility checks so schema updates that remove or weaken existing fields are rejected or clearly surfaced as errors.
- [x] T032 [P] [US2] Add tests that simulate an optimization loop: update the schema, re-run `StructureOfThought`, and verify that both primary outputs and `reasoning_json` respect the new schema while preserving earlier fields.
- [x] T033 [P] [US2] Add a focused example (or extend `dspy_so_example.py`) to show a small iterative refinement of `reasoning_json_schema` and how `reasoning_json` evolves while remaining a superset of the output structure.

**Checkpoint**: Schema evolution and extend-only compatibility are implemented and verified; optimization workflows can safely adjust `reasoning_json_schema`.

---

## Phase 5: User Story 3 - Use structured results in downstream tools (Priority: P3)

**Goal**: Make it easy for downstream tools to consume `StructuredResult` (including `reasoning_json`) without per-program parsing logic.

**Independent Test**: A mock evaluator or utility can be written once against the `StructuredResult` shape and reused across multiple `StructureOfThought` programs.

### Implementation for User Story 3

- [ ] T040 [US3] Add small helper utilities (if needed) in `dspy/predict/structure_of_thought.py` for accessing common fields and reasoning traces from `StructuredResult`.
- [ ] T041 [P] [US3] Add example usage in `examples/dspy_so_example.py` (or a nearby example file) that shows a downstream evaluation or logging routine consuming `result.answer`, `result.confidence`, and `result.reasoning_json`.
- [ ] T042 [P] [US3] Add tests in `tests/predict/test_structure_of_thought.py` that model a simple downstream consumer and confirm it works across multiple signatures and schemas without change.

**Checkpoint**: Downstream usage patterns are documented and tested; `StructuredResult` feels stable and convenient for consumers.

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Final improvements that cut across all user stories.

- [ ] T050 [P] [Polish] Review and refine docstrings and in-code comments in `dspy/predict/structure_of_thought.py` to match DSPy documentation standards (style, clarity, examples), building on the design notes added in Phase 2.
- [ ] T051 [P] [Polish] Ensure `quickstart.md` and `contracts/structure_of_thought_contracts.md` match the final implemented API.
- [ ] T052 [P] [Polish] Run the full test suite (including existing `tests/predict` and examples) and fix any regressions introduced by `StructureOfThought`.
- [ ] T053 [Polish] Consider minor performance and logging refinements (e.g., avoid logging overly large `reasoning_json` objects while still providing useful debugging context).

---

## Dependencies & Execution Order

- **Phase 1 (Setup)**: No dependencies; establishes files and imports.
- **Phase 2 (Foundational)**: Depends on Phase 1; blocks all user stories.
- **Phase 3 (US1)**: Depends on Phase 2; can be delivered as the MVP. After this phase, `StructureOfThought` behaves as a standard DSPy program and can already be optimized with existing teleprompters/proposers (e.g., `MIPROv2`, `GroundedProposer`) without any changes to those components.
- **Phase 4 (US2)**: Depends on Phase 2; can start after US1’s core path is stable but is logically independent in code structure. This phase strengthens the story around schema evolution (e.g., extend-only compatibility for `reasoning_json_schema`) but is not required to begin using teleprompting/optimization.
- **Phase 5 (US3)**: Depends on at least US1; can be done in parallel with portions of US2.
- **Phase N (Polish)**: Depends on desired user stories being complete.

Within each phase, tasks marked **[P]** can be done in parallel when working in different files with no direct dependencies.
