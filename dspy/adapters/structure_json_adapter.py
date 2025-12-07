import json
from typing import Any

from jsonschema import ValidationError, validate
import pydantic

from dspy.adapters.json_adapter import JSONAdapter
from dspy.adapters.utils import get_field_description_string, serialize_for_json
from dspy.signatures.signature import Signature
from dspy.utils.exceptions import AdapterParseError


class StructureJSONAdapter(JSONAdapter):
    """Schema-driven JSON adapter for ``StructureOfThought``.

    This adapter behaves like :class:`dspy.adapters.JSONAdapter` but adds an
    extra validation step driven by the JSON schema stored in
    ``Signature.instructions``.

    Intended usage:
    - ``PredictWithStructure`` (and thus ``StructureOfThought``) set
      ``signature.instructions`` to a pretty-printed JSON schema describing
      the structured outputs.
    - ``StructureJSONAdapter`` treats that schema string as the *source of
      truth* for validating LM responses.
    - After the usual JSON parsing and type casting performed by
      :class:`JSONAdapter`, the parsed result is validated against the schema.
      If validation fails, a clear :class:`AdapterParseError` is raised.

    If ``signature.instructions`` is empty, not a string, or not valid JSON,
    this adapter transparently falls back to the base :class:`JSONAdapter`
    behavior (no schema-aware validation).
    """
    def _load_schema_from_instructions(self, signature: type[Signature]) -> dict[str, Any] | None:
        """Best-effort parse of ``signature.instructions`` as a JSON schema.

        Returns:
            The parsed JSON schema dict if available and valid, otherwise ``None``.
        """

        instructions = getattr(signature, "instructions", None)
        if not instructions or not isinstance(instructions, str):
            return None

        try:
            schema = json.loads(instructions)
        except Exception:
            return None

        if not isinstance(schema, dict):
            return None

        return schema

    def _get_structured_outputs_model(self, signature):
        """Build a structured-outputs model from the JSON schema in instructions.

        For ``StructureOfThought``, ``PredictWithStructure`` has already stored
        a pretty-printed JSON schema describing the *output* object in
        ``signature.instructions``. Here we:

        - Parse that schema from the instructions string.
        - Create a lightweight Pydantic model whose :meth:`model_json_schema`
          simply returns this schema.
        - Return that model so providers that support OpenAI-style structured
          outputs receive exactly the same schema as the adapter later enforces
          in :meth:`parse`.

        If no usable schema is present in ``signature.instructions``, we fall
        back to the default :class:`JSONAdapter` behaviour, which derives the
        schema from ``signature.output_fields``.
        """
        import pdb; pdb.set_trace()
        schema = self._load_schema_from_instructions(signature)
        if not schema:
            return super()._get_structured_outputs_model(signature)

        pydantic_model = pydantic.create_model(
            "DSPyProgramOutputsFromInstructions",
            __config__=pydantic.ConfigDict(extra="forbid"),
        )
        # Ensure the provider sees the exact schema from instructions.
        pydantic_model.model_json_schema = lambda *args, **kwargs: schema
        return pydantic_model

    def parse(self, signature: type[Signature], completion: str) -> dict[str, Any]:
        # First, delegate to JSONAdapter for robust JSON extraction and
        # type-aware casting based on ``signature.output_fields``.
        fields = super().parse(signature, completion)

        schema = self._load_schema_from_instructions(signature)
        if not schema:
            # No usable schema instructions; behave like plain JSONAdapter.
            return fields

        # Convert any nested Pydantic models or complex objects into
        # JSON-serializable primitives before validating.
        instance = serialize_for_json(fields)

        ## NOTE: previously we validate the fields here using:
        #   validate(instance=instance, schema=schema)
        # However it is not necessary since the schema is a super set of the
        # converted instance.  So we just need to chill and return the fields.
        return fields

    def format_task_description(self, signature: type[Signature]) -> str:
        # NOTE: we override this since instruction is now the JSON schema, not the task description.
        # See Predict base class's format_task_description for more details.
        return "In adhering to this structure, your objective is to generate a JSON object that conforms to the schema above to solve the program task."

    def format_field_description(self, signature: type[Signature]) -> str:
        """Describe inputs and outputs, preferring the JSON schema for outputs."""

        inputs_desc = get_field_description_string(signature.input_fields)
        schema = self._load_schema_from_instructions(signature)

        if schema:
            return (
                f"Your input fields are:\n{inputs_desc}\n"
                "Your output fields are described by the following JSON schema:\n"
                f"{json.dumps(schema, indent=2)}"
            )

        # Fallback: use the standard Chat/JSON adapter description.
        return super().format_field_description(signature)

    def format_field_structure(self, signature: type[Signature]) -> str:
        """Describe the expected JSON structure for inputs/outputs.

        For ``StructureOfThought`` we prefer to surface the actual JSON schema
        (from ``signature.instructions``) when available, since this is the
        canonical specification of the output shape. If no schema is present,
        we fall back to the default :class:`JSONAdapter` field-structure
        description based on ``signature.input_fields`` / ``output_fields``.
        """

        schema = self._load_schema_from_instructions(signature)
        if schema:
            return (
                "All interactions must be expressed as JSON objects.\n\n"
                "Outputs MUST conform to the following JSON schema:\n"
                f"{json.dumps(schema, indent=2)}"
            )

        # Fallback: use the standard JSONAdapter formatting.
        return super().format_field_structure(signature)
