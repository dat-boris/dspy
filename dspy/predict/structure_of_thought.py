import json
import logging
from typing import Any

import dspy
import pydantic
from dspy.adapters.structure_json_adapter import StructureJSONAdapter
from dspy.predict.predict import Predict
from dspy.primitives.module import Module
from dspy.signatures.signature import Signature, ensure_signature

logger = logging.getLogger(__name__)


class StructuredResult(pydantic.BaseModel):
    """Container for the outputs of a StructureOfThought program.

    This wraps the primary structured outputs plus an optional reasoning_json
    object that is intended to be a structured superset of the primary outputs.

    For ergonomics and backwards-compatibility with existing DSPy patterns,
    attribute access is proxied to the `outputs` mapping when the attribute
    name matches a key there, so code can continue to use dot-access like
    `result.answer` or `result.response`.
    """

    outputs: dict[str, Any]
    reasoning_json: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None

    def __getattr__(self, item: str) -> Any:
        # Fall back to outputs[...] when the attribute is not a model field.
        try:
            return super().__getattribute__(item)
        except AttributeError as exc:
            outputs = super().__getattribute__("outputs")
            if isinstance(outputs, dict) and item in outputs:
                return outputs[item]
            raise exc

    def __getitem__(self, item: str) -> Any:
        return self.outputs[item]

    def pretty(self) -> str:
        """Return a pretty-printed JSON representation of the full structured result."""

        return self.model_dump_json(indent=2)


def _validate_reasoning_superset(outputs: dict[str, Any], reasoning_json: dict[str, Any] | None) -> None:
    """Ensure reasoning_json is a structured superset compatible with outputs.

    A minimal compatibility check for now:
    - Every key present in outputs MUST also appear in reasoning_json.
    - If an output value is not None, the corresponding value in reasoning_json
      MUST have the same type (based on Python's isinstance).

    This is intentionally conservative and focused on catching clear violations;
    we can refine the rules as the schema story evolves.
    """

    if reasoning_json is None:
        # Nothing to validate; callers can choose whether reasoning_json is required.
        return

    missing_keys: list[str] = []
    type_mismatches: list[str] = []

    for key, value in outputs.items():
        if key not in reasoning_json:
            missing_keys.append(key)
            continue

        other_value = reasoning_json[key]
        # Skip strict type comparison when the primary value is None.
        if value is None:
            continue

        if not isinstance(other_value, type(value)):
            type_mismatches.append(key)

    if missing_keys or type_mismatches:
        details = []
        if missing_keys:
            details.append(f"missing keys in reasoning_json: {sorted(missing_keys)}")
        if type_mismatches:
            details.append(f"type mismatches for keys: {sorted(type_mismatches)}")
        message = "reasoning_json is not a superset compatible with outputs; " + "; ".join(details)
        raise ValueError(message)


def _ensure_pydantic_outputs(signature: Signature) -> None:
    """Ensure all output fields use Pydantic models for strict structured outputs."""

    for name, field in signature.output_fields.items():
        annotation = field.annotation
        if not isinstance(annotation, type) or not issubclass(annotation, pydantic.BaseModel):
            raise TypeError(
                "StructureOfThought requires all output fields to be Pydantic models. "
                f"Field '{name}' has annotation {annotation!r}, which is not a subclass of pydantic.BaseModel."
            )


def _check_extend_only_compatible(old: Signature, new: Signature) -> None:
    """Validate that `new` is an extend-only evolution of `old` for outputs.

    Rules:
    - All output field names present in `old` MUST also be present in `new`.
    - For each such field, the type in `new` MUST be the same as in `old`,
      or a subclass (for Pydantic models). This prevents weakening types
      while allowing extension through subclassing or adding new fields.
    """

    missing: list[str] = []
    type_mismatches: list[str] = []

    for name, old_field in old.output_fields.items():
        if name not in new.output_fields:
            missing.append(name)
            continue

        new_field = new.output_fields[name]
        old_ann = old_field.annotation
        new_ann = new_field.annotation

        if not isinstance(old_ann, type) or not isinstance(new_ann, type):
            type_mismatches.append(name)
            continue

        # Allow identical types or subclasses (for Pydantic-style extension).
        if not (new_ann is old_ann or issubclass(new_ann, old_ann)):
            type_mismatches.append(name)

    if missing or type_mismatches:
        details = []
        if missing:
            details.append(f"missing output fields: {sorted(missing)}")
        if type_mismatches:
            details.append(f"incompatible output field types for: {sorted(type_mismatches)}")
        raise ValueError(
            "New signature is not extend-only compatible with previous signature; " + "; ".join(details),
        )


def _build_output_schema(signature: Signature) -> dict[str, Any]:
    """Build a JSON schema focused on the signature's output fields.

    This is derived from the full Pydantic-generated schema, but restricted to
    the output fields so that it can be used as a concise, human-readable
    description of the structured result shape.
    """

    full_schema = signature.model_json_schema()
    properties = full_schema.get("properties", {}) or {}

    output_props: dict[str, Any] = {}
    for name in signature.output_fields.keys():
        if name in properties:
            # Make a shallow copy so we can drop DSPy-specific extras.
            prop = dict(properties[name])
            prop.pop("__dspy_field_type", None)
            prop.pop("title", None)
            prop.pop("prefix", None)
            prop.pop("desc", None)
            output_props[name] = prop

    output_schema: dict[str, Any] = {
        "type": "object",
        "properties": output_props,
    }

    # Filter the required list down to just outputs, if present.
    if "required" in full_schema:
        required_outputs = [name for name in full_schema["required"] if name in output_props]
        if required_outputs:
            output_schema["required"] = required_outputs

    # Preserve any referenced definitions ($defs), since output types may rely on them.
    if "$defs" in full_schema:
        output_schema["$defs"] = full_schema["$defs"]

    return output_schema


class PredictWithStructure(Predict):
    """Predict primitive that exposes its output schema via `Signature.instructions`.

    Responsibilities:
    - Enforce that all outputs are Pydantic models (strict structured outputs).
    - Compute a JSON schema describing the output structure (including any
      `reasoning_json`) from the Signature.
    - Store this schema as `output_schema` / `output_schema_json`.
    - Set `self.signature.instructions` to the pretty-printed JSON schema so
      that teleprompters/proposers can read it via::

          basic_instruction = get_signature(predictor).instructions

    In other words, for modules built from `PredictWithStructure`, the
    \"instruction\" for a predictor *is* the JSON schema of the structured
    outputs (rather than a natural-language prompt), which downstream tools
    can use when proposing updated schemas or reasoning structures.
    """

    def __init__(self, signature: str | type[Signature], callbacks=None, **config: dict[str, Any]):
        super().__init__(signature=signature, callbacks=callbacks, **config)

        # Enforce and derive schema from the bound signature.
        _ensure_pydantic_outputs(self.signature)
        self.output_schema: dict[str, Any] = _build_output_schema(self.signature)
        self.output_schema_json: str = json.dumps(self.output_schema, indent=2, sort_keys=True)

        # Expose the schema via instructions so optimizers/proposers can see it.
        self.signature.instructions = self.output_schema_json

    def forward(self, **kwargs):
        """Call the LM using a schema-driven JSON adapter.

        This ensures that, for modules built on top of ``PredictWithStructure``,
        LM responses are validated against the JSON schema stored in
        ``signature.instructions`` (as produced by this class or updated by a
        teleprompter/optimizer).
        """

        from dspy.dsp.utils.settings import settings

        with settings.context(adapter=StructureJSONAdapter()):
            return super().forward(**kwargs)

    async def aforward(self, **kwargs):
        """Async counterpart of :meth:`forward` using ``StructureJSONAdapter``."""

        from dspy.dsp.utils.settings import settings

        with settings.context(adapter=StructureJSONAdapter()):
            return await super().aforward(**kwargs)


# This is the description that will be used by proposer to use as fht module description
# when generating the instruction for the module.
STRUCTURE_OF_THOUGHT_MODULE_DESCRIPTION = """
This module uses the `proposed_instruction` (a JSON schema) to decide the structure of the model's output when solving the task described in `program_description`. The previous instruction is provided in `basic_instruction`, and your goal is to produce an improved `proposed_instruction` by rewriting and optimizing the JSON schema—while ensuring it remains fully compatible with the `baseline_schema` defined below.

Your output must be a valid JSON Schema. It must remain a superset of the baseline schema. You may add new fields, reorder fields, strengthen descriptions, expand reasoning structures, and introduce optional metadata fields. You may not remove or weaken any required fields from the baseline.

----------------------------------------------------------------------
Objective
----------------------------------------------------------------------

Rewrite and enhance a JSON Schema so that the model produces more structured reasoning, clearer intermediate states, and a well-ordered final answer.

The task context is provided in:

    program_description

Update the schema with a focus on:
- Improving field descriptions
- Reordering fields to influence output order
- Strengthening field naming and descriptions to explicitly guide expected content
- Expanding reasoning into multi-step structures
- Adding metadata to mitigate common LLM reasoning failures
- Ensuring the entire output remains valid JSON Schema

----------------------------------------------------------------------
Guidelines
----------------------------------------------------------------------

1. Field Ordering
Place reasoning or intermediate-state fields before the final answer field to enforce preferred generation order.

Undesired:
{
  "answer": "string",
  "reasoning": "string"
}

Preferred:
{
  "reasoning": "string",
  "answer": "string"
}

2. Expand Reasoning Structures
Replace simple reasoning strings with structured or multi-step representations, such as:

{
  "steps": [
    {
      "explanation": "string",
      "current_state": "string",
      "next_action": "string"
    }
  ],
  "answer": "string"
}

Optional improvements may include:
- hypotheses or alternative branches
- state snapshots and transitions
- uncertainty annotations
- validation passes
- dead-end tracking or missteps

3. Field Naming and Descriptions (Very Important)
Carefully choose clear and directive field names and descriptions. The naming and description of each field strongly influences what the LLM places in that field.

For example:
- A field named `"detailed_reasoning_steps"` with the description *"Each step must explicitly reference the current state and the logical inference made"* produces better structured reasoning than a vague `"reasoning"` field.
- A field named `"final_numeric_answer"` or `"chosen_option_letter"` informs the model of the expected answer type.
- Descriptions should define format, purpose, constraints, and expectations (e.g., “one sentence,” “list of deductions,” “explicit variables used,” etc.).

You are encouraged to redesign and rename fields whenever it improves clarity and guidance, as long as compatibility with the baseline schema is maintained.

4. Add Creative but Valid Optional Fields
Additional optional fields may improve reliability, for example:

"self_checks": [
  { "check": "string", "result": "string" }
]

"missteps": [
  { "description": "string" }
]

These fields help the model avoid common failure modes such as skipping steps or hallucinating unavailable information.

5. Maintain JSON Schema Validity
All outputs must strictly follow JSON Schema rules. The final output must contain only the schema itself, without commentary.

----------------------------------------------------------------------
Baseline Schema (Your Output Must Be a Superset)
----------------------------------------------------------------------

```json
{baseline_schema}
```

----------------------------------------------------------------------
Required Output Format (Strict)
----------------------------------------------------------------------

Return only a JSON Schema that is a valid superset of the baseline schema. No prose, no explanation—only the schema.
"""


class StructureOfThought(Module):
    """StructureOfThought program.

    Design goals:
    - Behave like a standard DSPy program from the perspective of teleprompters
      and proposers (e.g., MIPROv2, GroundedProposer) by exposing predictors
      that are subclasses of `dspy.Predict` and therefore visible to
      `module.predictors()`, `get_signature(predictor)`, etc.
    - Provide a strongly structured result via `StructuredResult`, including
      optional `reasoning_json` that is a superset-compatible trace of the
      primary outputs.
    - Maintain an explicit JSON schema of the output structure (including any
      `reasoning_json`) and use that schema as the canonical description of
      what the program returns.

    Critical interaction with teleprompters/proposers:
    - Components like `GroundedProposer` obtain a "basic instruction" for each
      predictor via::

          basic_instruction = get_signature(program.predictors()[pred_i]).instructions

    - `StructureOfThought` therefore composes `PredictWithStructure` predictors,
      which automatically set their `Signature.instructions` string to a
      pretty-printed JSON schema of the output structure. This means that
      `basic_instruction` becomes the JSON schema text for the outputs, giving
      proposers direct access to the structured result shape when generating or
      refining instructions/schemas.
    - When `rebind_signature` is used in optimization flows, `StructureOfThought`
      MUST regenerate both:
        * the stored output JSON schema, and
        * the underlying `PredictWithStructure` instances (and thus their
          `Signature.instructions` strings)
      so that teleprompters/proposers always see an up-to-date schema when they
      call `get_signature(predictor).instructions`.
    """

    def __init__(self, signature: str | type[Signature], *, strict: bool = True, **config: dict[str, Any]):
        super().__init__()
        self.signature: Signature = ensure_signature(signature)
        self.strict: bool = strict
        self._predict_config: dict[str, Any] = dict(config)

        # Enforce that all outputs are structured via Pydantic models.
        _ensure_pydantic_outputs(self.signature)

        # Cache a JSON schema focused on output fields for easy inspection.
        self.output_schema: dict[str, Any] = _build_output_schema(self.signature)
        self.output_schema_json: str = json.dumps(self.output_schema, indent=2, sort_keys=True)

        # Use PredictWithStructure so predictors expose JSON schema via instructions
        # while still behaving like standard Predict instances to the rest of DSPy.
        self._predict = PredictWithStructure(self.signature, **self._predict_config)


    @property
    def module_description(self) -> str:
        # Use the current Pydantic-derived output schema as the baseline.
        # NOTE: we use replace here as `.format()` is more generic and could cause KeyError
        # if the prompt template includes other placeholders.
        return STRUCTURE_OF_THOUGHT_MODULE_DESCRIPTION.replace("{baseline_schema}", self.output_schema_json)

    def schema_repr(self) -> str:
        """Return a human-readable JSON schema for the structured outputs.

        This is equivalent to `output_schema_json` and focuses on the output
        side of the signature, not inputs, for easier reasoning about result
        structure.
        """

        return self.output_schema_json

    def rebind_signature(self, signature: str | type[Signature]) -> None:
        """Update the bound signature in an extend-only compatible way.

        This is intended for optimization/teleprompting-style workflows that
        evolve the schema over time. It:
        - Ensures new outputs remain Pydantic models.
        - Enforces extend-only compatibility with the previous signature.
        - Rebuilds the underlying Predict module with the new signature while
          preserving its configuration.
        """

        new_sig = ensure_signature(signature)
        _ensure_pydantic_outputs(new_sig)
        _check_extend_only_compatible(self.signature, new_sig)

        self.signature = new_sig
        self.output_schema = _build_output_schema(self.signature)
        self.output_schema_json = json.dumps(self.output_schema, indent=2, sort_keys=True)
        self._predict = PredictWithStructure(self.signature, **self._predict_config)

    def forward(self, **kwargs):
        # Phase 3: delegate to the underlying predictor, then wrap in StructuredResult.
        raw_prediction = self._predict(**kwargs)

        # Extract primary outputs as a plain dict.
        outputs: dict[str, Any] = {}
        for name in self.signature.output_fields.keys():
            if hasattr(raw_prediction, name):
                outputs[name] = getattr(raw_prediction, name)
            elif isinstance(raw_prediction, dict) and name in raw_prediction:
                outputs[name] = raw_prediction[name]

        # Optional: treat a dedicated `reasoning_json` field as the richer reasoning structure.
        reasoning_json: dict[str, Any] | None = None
        if "reasoning_json" in outputs and isinstance(outputs["reasoning_json"], dict):
            reasoning_json = outputs.pop("reasoning_json")

        if self.strict:
            _validate_reasoning_superset(outputs, reasoning_json)

        return StructuredResult(outputs=outputs, reasoning_json=reasoning_json)

    async def aforward(self, **kwargs):
        return await self._predict.acall(**kwargs)


