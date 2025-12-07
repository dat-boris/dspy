import json
import logging
from typing import Any

import dspy
import pydantic
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


class StructureOfThought(Module):
    """StructureOfThought program (Phase 1–2 implementation).

    Current behavior:
    - Wraps a standard dspy.Predict constructed from the provided signature so
      it already fits into existing teleprompting/optimization flows that
      expect a normal DSPy program.
    - Introduces a `StructuredResult` container type and a minimal superset
      validator `_validate_reasoning_superset` that will be used in later
      phases when we attach explicit reasoning_json handling to LM outputs.

    Later phases will:
    - Bind a signature that includes both output schema and reasoning_json_schema.
    - Invoke the underlying LM via existing DSPy prediction primitives.
    - Return strictly validated StructuredResult instances.
    """

    def __init__(self, signature: str | type[Signature], *, strict: bool = True, **config: dict[str, Any]):
        super().__init__()
        self.signature: Signature = ensure_signature(signature)
        self.strict: bool = strict

        # Enforce that all outputs are structured via Pydantic models.
        for name, field in self.signature.output_fields.items():
            annotation = field.annotation
            if not isinstance(annotation, type) or not issubclass(annotation, pydantic.BaseModel):
                raise TypeError(
                    "StructureOfThought requires all output fields to be Pydantic models. "
                    f"Field '{name}' has annotation {annotation!r}, which is not a subclass of pydantic.BaseModel."
                )

        # Reuse the existing Predict primitive so we immediately behave like a
        # standard DSPy program from the perspective of teleprompters/proposers.
        self._predict = dspy.Predict(self.signature, **config)

    def schema_repr(self) -> str:
        """Return a human-readable JSON schema for the bound signature."""

        schema = self.signature.model_json_schema()
        return json.dumps(schema, indent=2, sort_keys=True)

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


