# Feature Specification: StructureOfThought Module

**Feature Branch**: `002-structure-of-thought`
**Created**: 2025-12-07
**Status**: Draft
**Input**: User description: "Create a new module, called `StructureOfThought`, where we can create the program via `program = dspy.StructureOfThought(ExampleSignature)`. The program output must be a structured output (using a schema/validation library), and the output structure can be optimized as the system updates the signature and schema (similar in spirit to how MIPROv2 updates signatures). For reference, existing `ChainOfThought` and `ProgramOfThought` patterns should inform the design."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Define a structured reasoning program (Priority: P1)

A DSPy practitioner wants to define a new reasoning program that always returns a well-structured result (for example, fields like "answer" and "confidence"), together with a richer `reasoning_json` object whose shape is described by a `reasoning_json_schema`. The `reasoning_json` is itself structured (not free-form text) and is designed to be a superset that remains compatible with the primary output structure.

**Why this priority**: This is the core value of the feature: making it easy and safe to obtain structured outputs from reasoning programs so they can be consumed by downstream tools and evaluations without custom parsing.

**Independent Test**: A user can define a simple signature, wrap it with `StructureOfThought`, run it on an example input, and receive a consistently structured object that can be accessed via fields (e.g., `result.answer`, `result.reasoning_json`) across multiple runs, with `reasoning_json` validated against the declared `reasoning_json_schema`.

**Acceptance Scenarios**:

1. **Given** a signature that specifies fields for the answer and explanation, **When** the user constructs a `StructureOfThought` program and runs it on valid inputs, **Then** the system returns a result object whose fields match the signature and can be accessed without manual parsing.
2. **Given** a user who is familiar with existing DSPy programs (e.g., `ProgramOfThought`), **When** they switch to `StructureOfThought` using a similar call pattern, **Then** they can get structured results without learning a completely new mental model.

---

### User Story 2 - Evolve schemas during optimization (Priority: P2)

A teleprompting or optimization workflow (such as a MIPRO-style optimizer) wants to adjust the instructions and output schema of a program over time, while keeping the user-facing API stable and continuing to return structured objects.

**Why this priority**: Many advanced users tune instructions and schemas iteratively; `StructureOfThought` should support these workflows so optimized prompts still yield valid structured outputs.

**Independent Test**: An optimization routine can modify the signature/schema used by `StructureOfThought` across iterations, and each resulting program invocation still produces structured outputs that match the latest schema.

**Acceptance Scenarios**:

1. **Given** an optimization loop that updates the signature or schema between iterations, **When** a new `StructureOfThought` program is constructed or updated, **Then** the program’s outputs conform to the new schema without breaking existing calling code.
2. **Given** a previously stored result from an earlier schema version, **When** the schema is updated, **Then** users are responsible for handling any compatibility or migration of past results, while `StructureOfThought` focuses on ensuring that the current schema remains compatible with the original field set (i.e., previously declared fields and types are preserved or extended, not removed or weakened).

---

### User Story 3 - Use structured results in downstream tools (Priority: P3)

A downstream tool (e.g., an evaluator, retriever, or application) consumes outputs from `StructureOfThought` and relies on strong guarantees about field names and types to make decisions, log metrics, or store data.

**Why this priority**: The main benefit of structured reasoning is simpler, safer integration into larger systems that depend on predictable fields and shapes.

**Independent Test**: A downstream component can be written once against the structured interface (e.g., expecting `answer`, `supporting_facts`, `metadata` fields) and used across multiple `StructureOfThought` programs without per-program parsing logic.

**Acceptance Scenarios**:

1. **Given** a `StructureOfThought` program that returns fields such as `answer` and `supporting_facts`, **When** a downstream evaluator is configured to read those fields, **Then** the evaluator can run across many examples without encountering missing or renamed fields.
2. **Given** a user who changes the internal instructions of `StructureOfThought` while keeping the schema the same, **When** the program is re-run, **Then** downstream tools continue to work without modifications because the structured interface is unchanged.

---

### Edge Cases

- What happens when model output cannot be parsed into the expected structured format (e.g., missing fields, wrong types, or malformed content)? By default, the system raises a clear, structured error and does not return a structured result object, so failures are explicit and easy to detect.
- How does the system handle schema changes that remove or rename fields which downstream tools might still expect? (e.g., guidance on deprecation, explicit errors, or compatibility helpers).
- How are extremely large or deeply nested structured outputs handled in terms of performance, logging, and inspection (e.g., truncation, summarized views, or limits on depth/size)?

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST allow users to define a reasoning program using a concise `StructureOfThought` constructor that accepts a signature describing the desired structured output fields and an associated `reasoning_json_schema` that governs the shape of a richer `reasoning_json` object.
- **FR-002**: System MUST ensure that, on successful completion, each invocation of a `StructureOfThought` program returns a structured result object whose fields and basic types are consistent with the provided signature/schema, and whose `reasoning_json` (when present) conforms to the declared `reasoning_json_schema`.
- **FR-003**: Users MUST be able to access structured fields on the result object via attribute or key-style access without needing to manually parse raw text.
- **FR-004**: System MUST provide clear, human-readable feedback when a structured result cannot be produced (for example, due to invalid model output or schema mismatch), so users can debug and adjust their signatures or instructions.
- **FR-005**: System MUST support integration into existing DSPy workflows (including teleprompting/optimization), such that `StructureOfThought` can be used wherever a standard program object is expected, without requiring special-case handling. In particular, it MUST work with existing teleprompters and proposers (e.g., `MIPROv2`, `GroundedProposer`) purely by conforming to current program/signature conventions (predictors, `get_signature`/`set_signature`, `instructions`, `fields`), without requiring changes to those components in the initial version.
- **FR-006**: System MUST support updating or replacing the signature/schema associated with a `StructureOfThought` program (for example, within an optimization loop), and MUST ensure subsequent outputs conform to the updated schema while keeping newly introduced schemas compatible with the original declared fields and types (i.e., new schemas may extend but should not break previously defined fields).
- **FR-007**: System MUST provide a way for advanced users to inspect the effective schema for a given `StructureOfThought` program (for example, via a representation or helper function) so they can verify the expected structure before running large jobs.
- **FR-008**: System MUST define and document a default behavior for handling partial or invalid structured outputs: by default, the system raises a clear, structured error and does not return a structured result object when the output cannot be validated against the schema.
- **FR-009**: System MUST avoid leaking low-level implementation details (such as specific validation libraries) into the user-facing API, focusing instead on stable concepts like "schema", "fields", and "structured result".
- **FR-010**: When exposing or logging the structured output schema (or example structured outputs), the system MUST present them in a human-readable, properly indented format (e.g., pretty-printed JSON-like structures) to support easy manual inspection and debugging.

### Key Entities *(include if feature involves data)*

- **StructureOfThought Program**: A reasoning program constructed from a signature that, when invoked, produces a structured result object rather than unstructured text. Key attributes include the associated signature/schema and any configuration options that affect how structure is enforced.
- **Output Schema / Signature**: The declared shape of the structured result, including field names, basic types, and optional/required status. This may evolve over time (e.g., during optimization), but at any given moment, each `StructureOfThought` program has a well-defined active schema.
- **Structured Result Object**: The object returned by `StructureOfThought` invocations, exposing fields in a predictable way and suitable for use by downstream components, logging, and evaluation.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Practitioners can define and successfully run a new structured reasoning program (from initial signature to first structured result) in under 10 minutes, following only high-level documentation.
- **SC-002**: In internal or example workflows that previously required custom parsing of free-form reasoning outputs, at least 80% of those workflows can be migrated to use `StructureOfThought` with no custom parsing logic.
- **SC-003**: In a representative benchmark or internal evaluation, at least 95% of `StructureOfThought` invocations that complete successfully produce structured results that fully conform to the declared schema (no missing or mis-typed fields).
- **SC-004**: At least 90% of users in a small usability study (or internal team review) report that they understand how to define the structure of outputs and how to consume the resulting structured objects after reading the feature documentation once.
