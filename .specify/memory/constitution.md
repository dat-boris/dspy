<!--
Sync Impact Report:
Version change: N/A → 1.0.0 (initial constitution)
Modified principles: N/A (new document)
Added sections: Core Principles (8 principles), Development Standards, Code Quality Requirements, Governance
Templates requiring updates:
  ✅ plan-template.md - Constitution Check section aligns with principles
  ✅ spec-template.md - Requirements section aligns with type safety and testing principles
  ✅ tasks-template.md - Task organization aligns with modular architecture and testing principles
Follow-up TODOs: None
-->

# DSPy Constitution

## Core Principles

### I. Modular Architecture (NON-NEGOTIABLE)
All components MUST inherit from `dspy.Module` to maintain consistent interface and composability. Modules compose other modules rather than relying on deep inheritance hierarchies. Each module MUST have a clear, single responsibility. The base `Module` class provides essential infrastructure (history tracking, callbacks, usage tracking) that all modules inherit.

**Rationale**: Modular architecture enables code reuse, testability, and maintainability. The consistent interface allows optimizers and tools to work uniformly across all DSPy components.

### II. Type Safety and Annotations
All public APIs MUST use comprehensive type hints. Use modern Python typing (`str | type[Signature]`, `dict[str, Any]`, etc.). Forward references MUST use `TYPE_CHECKING` guards. Pydantic models MUST be used for data validation where appropriate. Type checking MUST pass before code is merged.

**Rationale**: Type hints improve code clarity, enable better IDE support, catch errors early, and make the codebase more maintainable as it scales.

### III. Adapter Pattern for LM Abstraction
Language Model interactions MUST be abstracted through the `Adapter` base class. Specialized adapters (`ChatAdapter`, `JSONAdapter`) handle different LM interfaces. Adapters MUST implement graceful fallback mechanisms (e.g., `ChatAdapter` falls back to `JSONAdapter` on failure). Adapters MUST handle both sync and async operations consistently.

**Rationale**: The adapter pattern allows DSPy to work with diverse LM providers while maintaining a consistent programming model. Fallbacks ensure robustness when LM interfaces change or fail.

### IV. Comprehensive Error Handling
Custom exceptions MUST provide detailed context (adapter name, signature, LM response, parsed results). Error messages MUST be actionable and include sufficient debugging information. Critical operations MUST have try/except blocks with appropriate fallback behavior. Exceptions MUST inherit from appropriate base classes (`AdapterParseError` for parsing failures).

**Rationale**: Detailed error handling reduces debugging time and improves user experience. Context-rich exceptions help developers understand and fix issues quickly.

### V. Documentation Standards
All public classes and methods MUST have docstrings following Google-style format. Docstrings MUST include parameter descriptions, return value descriptions, and usage examples where appropriate. Complex algorithms MUST include inline comments explaining non-obvious logic. README files MUST be kept up-to-date with installation and usage instructions.

**Rationale**: Good documentation enables contributors to understand and extend the codebase. Examples in docstrings serve as both documentation and tests.

### VI. Async Support
All modules that perform I/O operations MUST provide both synchronous (`forward`, `__call__`) and asynchronous (`aforward`, `acall`) interfaces. Async implementations MUST use proper async/await patterns and avoid blocking operations. Both interfaces MUST produce equivalent results.

**Rationale**: Async support enables efficient concurrent operations and better resource utilization, especially important for LM API calls and parallel processing.

### VII. Logging and Observability
All modules MUST use module-level loggers: `logger = logging.getLogger(__name__)`. Logging MUST use appropriate levels (ERROR for failures, WARNING for recoverable issues, INFO for important events). History tracking MUST be built into modules for debugging and optimization. Usage tracking (token counting, etc.) MUST be integrated where applicable.

**Rationale**: Observability is critical for debugging production issues and understanding system behavior. History tracking enables prompt optimization and debugging.

### VIII. Testing Discipline
All new features MUST include corresponding tests. Tests MUST be organized by module/feature area. Unit tests MUST cover core functionality. Integration tests MUST verify module composition. Test coverage SHOULD be maintained at high levels. Tests MUST run in CI/CD pipeline before merge.

**Rationale**: Comprehensive testing ensures code quality, prevents regressions, and enables confident refactoring. Tests serve as executable documentation.

## Development Standards

### Code Style
Code MUST follow the Google Python Style Guide. Formatting and linting MUST be enforced via `ruff`. Pre-commit hooks MUST be configured to run `ruff` automatically. All code MUST pass linting checks before commit.

### Configuration Management
Global configuration MUST use the singleton `Settings` class pattern. Thread-safe configuration with thread-local overrides MUST be supported for parallel execution. Context managers MUST be used for temporary configuration changes. Configuration changes MUST be documented.

### Extensibility
Callback system MUST be available for extending module behavior. Decorators (`with_callbacks`) MUST be used to integrate callbacks automatically. New features SHOULD be designed to be extensible via callbacks or subclassing rather than requiring core changes.

### Metaclasses and Advanced Patterns
Metaclasses MAY be used for advanced initialization patterns (e.g., `ProgramMeta` ensuring proper module initialization). Metaclass usage MUST be justified and documented. Base initialization MUST occur even if subclasses forget `super().__init__()`.

## Code Quality Requirements

### Composition Over Inheritance
Prefer composition of modules over deep inheritance hierarchies. Modules SHOULD compose other modules to build complex functionality. Inheritance SHOULD be used primarily for establishing base contracts (e.g., `Module`, `Adapter`).

### Defensive Programming
Input validation MUST occur at module boundaries. Output validation MUST verify results match expected signatures. Edge cases MUST be handled gracefully. Required attributes MUST be checked and initialized with sensible defaults.

### String-Based Signatures
Support for string-based signature definitions (`"input -> output"`) MUST be maintained for ease of use. Type-safe class-based signatures MUST be the preferred approach for complex signatures. `ensure_signature()` MUST handle both string and class-based inputs flexibly.

### History and Usage Tracking
Modules MUST maintain execution history for debugging and optimization. Token usage tracking MUST be integrated where applicable. Inspection methods (`inspect_history`) MUST be available for debugging.

## Governance

### Constitution Authority
This constitution supersedes all other coding practices and style guides. All PRs and code reviews MUST verify compliance with these principles. Violations MUST be justified in PR descriptions with rationale for exceptions.

### Amendment Process
Amendments to this constitution require:
1. Documentation of the proposed change and rationale
2. Review and approval by maintainers
3. Update to version number following semantic versioning:
   - **MAJOR**: Backward incompatible principle removals or redefinitions
   - **MINOR**: New principles or materially expanded guidance
   - **PATCH**: Clarifications, wording fixes, non-semantic refinements
4. Propagation of changes to dependent templates and documentation

### Compliance Review
All code contributions MUST be reviewed for constitution compliance. Reviewers MUST flag violations and request justification or fixes. Complexity additions MUST be justified with rationale for why simpler alternatives were insufficient.

### Version Control
Constitution changes MUST be committed with clear messages indicating version bump and nature of changes. Example: `docs: amend constitution to v1.1.0 (add async support principle + governance update)`

**Version**: 1.0.0 | **Ratified**: 2025-01-27 | **Last Amended**: 2025-01-27
