# Data Model: StructureOfThought Module

## Entities

### 1. StructureOfThought Program

Represents a reasoning program that, when invoked, produces a structured result object rather than free-form text.

- **Key attributes**:
  - Associated signature or schema definition (fields, types, optional/required flags).
  - Configuration flags (e.g., strict vs future tolerant modes, though strict is the default).
  - References to underlying prediction primitives and adapters.
- **Relationships**:
  - Composes existing prediction modules (e.g., `ProgramOfThought`-like components) and adapters.
  - Produces `StructuredResult` instances according to an `OutputSchema`.

### 2. OutputSchema / Signature

Describes the expected shape of the structured result.

- **Key attributes**:
  - Field names and basic types (e.g., `answer: str`, `reasoning_steps: list[str]`, `confidence: float | None`).
  - Optional/required flags and any simple constraints (e.g., non-empty lists, bounded numeric ranges when applicable).
  - Extend-only compatibility rule: new schemas may add fields but should not remove or weaken previously declared ones.
- **Relationships**:
  - Bound to a `StructureOfThought` program instance.
  - May be updated over time (e.g., by optimization workflows), subject to compatibility rules.

### 3. StructuredResult

The object returned by a `StructureOfThought` invocation.

- **Key attributes**:
  - Fields defined in the active `OutputSchema` (e.g., `answer`, `reasoning_steps`, `supporting_facts`, `metadata`).
  - Validation status and potential diagnostic metadata (e.g., for logging/history), while still keeping strict behavior as default.
- **Relationships**:
  - Produced by `StructureOfThought` for each call.
  - Consumed by downstream tools (evaluators, retrievers, applications).

## Validation Rules

- Structured results MUST conform to the active `OutputSchema`: missing required fields, wrong types, or malformed content cause a validation error.
- Schema updates MUST preserve previously declared fields and their basic types to remain compatible; breaking changes require user-managed migration.
- Pretty-printed representations of schemas and example results MUST be stable and deterministic to support debugging and diffing.

## State Transitions (High-Level)

1. **Schema binding**: A user defines a signature/schema and constructs a `StructureOfThought` program.
2. **Invocation**: The program runs, using LM adapters to generate candidate outputs and then parsing/validating them into a `StructuredResult`.
3. **Success**: If validation passes, a `StructuredResult` is returned and can be logged, inspected, and consumed.
4. **Failure**: If validation fails, a structured error is raised; history/logging record the failure context.
5. **Schema update**: An optimization or manual change updates the signature/schema; subsequent invocations use the new schema, subject to extend-only compatibility.
