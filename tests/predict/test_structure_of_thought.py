import json

import pytest

import dspy
import pydantic
from dspy.predict.structure_of_thought import (
    PredictWithStructure,
    StructuredResult,
    _check_extend_only_compatible,
    _ensure_pydantic_outputs,
    _validate_reasoning_superset,
)
from dspy.teleprompt.utils import get_signature
from dspy.utils import DummyLM
from dspy.utils.exceptions import AdapterParseError


def test_structure_of_thought_import_and_basic_instantiation_with_pydantic_output():
    class SimpleResponse(pydantic.BaseModel):
        answer: str

    # Define a signature whose output field is a Pydantic model.
    class SimpleSignature(dspy.Signature):
        question: str = dspy.InputField()
        response: SimpleResponse = dspy.OutputField()

    sot = dspy.StructureOfThought(SimpleSignature)

    # Basic sanity checks that it behaves like a DSPy module.
    assert isinstance(sot, dspy.Module)
    assert sot.signature is not None
    assert "question" in sot.signature.input_fields
    assert "response" in sot.signature.output_fields

    # The underlying predictor should be a PredictWithStructure.
    predictors = sot.predictors()
    assert len(predictors) == 1
    predictor = predictors[0]
    assert isinstance(predictor, PredictWithStructure)

    # Its instructions should contain a JSON schema describing the outputs.
    sig = get_signature(predictor)
    schema = json.loads(sig.instructions)
    assert "properties" in schema
    assert "response" in schema["properties"]

    # We cannot call the LM in tests, but we can still construct a StructuredResult
    # and rely on its ergonomic access patterns.
    result = StructuredResult(outputs={"response": SimpleResponse(answer="42")})
    assert isinstance(result.response, SimpleResponse)
    assert result.response.answer == "42"
    assert result.outputs["response"].answer == "42"


def test_structured_result_container_and_superset_validator_success():
    outputs = {"answer": "42", "confidence": 0.9}
    reasoning_json = {"answer": "42", "confidence": 0.9, "extra_detail": "something"}

    # Should not raise for a compatible superset and should be constructible.
    _validate_reasoning_superset(outputs, reasoning_json)
    result = StructuredResult(outputs=outputs, reasoning_json=reasoning_json)
    assert result.outputs["answer"] == "42"
    assert "extra_detail" in result.reasoning_json


def test_superset_validator_raises_on_missing_or_mismatched_fields():
    outputs = {"answer": "42", "confidence": 0.9}
    # Missing 'confidence' and wrong type for 'answer'.
    reasoning_json = {"answer": 42}

    with pytest.raises(ValueError):
        _validate_reasoning_superset(outputs, reasoning_json)


def test_structure_of_thought_raises_if_output_not_pydantic_model():
    # Signature with a plain str output should be rejected.
    class BadSignature(dspy.Signature):
        question: str = dspy.InputField()
        answer: str = dspy.OutputField()

    with pytest.raises(TypeError):
        dspy.StructureOfThought(BadSignature)


def test_structure_of_thought_signature_and_forward_with_dummy_predict():
    class SimpleResponse(pydantic.BaseModel):
        answer: str

    class SimpleSignature(dspy.Signature):
        question: str = dspy.InputField()
        response: SimpleResponse = dspy.OutputField()

    # Use a dummy predictor to avoid real LM calls.
    class DummyPredict(dspy.Module):
        def forward(self, **kwargs):
            # Mimic what a normal Predict would return: an object with attributes
            # matching the signature's output fields.
            return dspy.Prediction(response=SimpleResponse(answer="42"))

    sot = dspy.StructureOfThought(SimpleSignature)
    # Inject the dummy predictor so no external calls are made.
    sot._predict = DummyPredict()

    result = sot(question="What is 6 * 7?")

    # The result should be a StructuredResult with ergonomic access.
    assert isinstance(result, StructuredResult)
    assert isinstance(result.response, SimpleResponse)
    assert result.response.answer == "42"
    # And we can pretty-print it.
    pretty = result.pretty()
    assert "{\n" in pretty


def test_structure_of_thought_schema_text_representation(capsys):
    class SimpleResponse(pydantic.BaseModel):
        answer: str

    class SimpleSignature(dspy.Signature):
        question: str = dspy.InputField()
        response: SimpleResponse = dspy.OutputField()

    adapter = dspy.JSONAdapter()
    text = adapter.format_field_structure(SimpleSignature)

    # Print the structure so it is easy to inspect when running tests manually.
    print(text)

    # Basic sanity check that the formatted structure mentions inputs and outputs.
    assert "Inputs will have the following structure" in text
    assert "Outputs will be a JSON object with the following fields" in text


def test_structure_of_thought_schema_repr_uses_pretty_json():
    class SimpleResponse(pydantic.BaseModel):
        answer: str

    class SimpleSignature(dspy.Signature):
        question: str = dspy.InputField()
        response: SimpleResponse = dspy.OutputField()

    sot = dspy.StructureOfThought(SimpleSignature)
    schema_text = sot.schema_repr()

    # Should be valid JSON and pretty-printed (multi-line with indentation) and
    # focused on output fields.
    parsed = json.loads(schema_text)
    assert isinstance(parsed, dict)
    assert "properties" in parsed
    assert set(parsed["properties"].keys()) == {"response"}
    assert "{\n" in schema_text


def test_ensure_pydantic_outputs_enforces_pydantic_models():
    class OkModel(pydantic.BaseModel):
        value: str

    class GoodSig(dspy.Signature):
        x: str = dspy.InputField()
        y: OkModel = dspy.OutputField()

    # Should not raise.
    _ensure_pydantic_outputs(GoodSig)

    class BadSig(dspy.Signature):
        x: str = dspy.InputField()
        y: str = dspy.OutputField()

    with pytest.raises(TypeError):
        _ensure_pydantic_outputs(BadSig)


def test_check_extend_only_compatible_allows_superset_and_subclass_types():
    class BaseOut(pydantic.BaseModel):
        value: str

    class ExtendedOut(BaseOut):
        extra: int | None = None

    class OldSig(dspy.Signature):
        x: str = dspy.InputField()
        y: BaseOut = dspy.OutputField()

    class NewSig(dspy.Signature):
        x: str = dspy.InputField()
        y: ExtendedOut = dspy.OutputField()
        z: ExtendedOut = dspy.OutputField()

    # NewSig adds a field and uses a subclass type for y; this is allowed.
    _check_extend_only_compatible(OldSig, NewSig)


def test_check_extend_only_compatible_rejects_missing_or_weakened_fields():
    class BaseOut(pydantic.BaseModel):
        value: str

    class DifferentOut(pydantic.BaseModel):
        other: int

    class OldSig(dspy.Signature):
        x: str = dspy.InputField()
        y: BaseOut = dspy.OutputField()

    class MissingFieldSig(dspy.Signature):
        x: str = dspy.InputField()

    class WeakenedTypeSig(dspy.Signature):
        x: str = dspy.InputField()
        y: DifferentOut = dspy.OutputField()

    with pytest.raises(ValueError):
        _check_extend_only_compatible(OldSig, MissingFieldSig)

    with pytest.raises(ValueError):
        _check_extend_only_compatible(OldSig, WeakenedTypeSig)


def test_rebind_signature_updates_signature_and_preserves_config():
    class BaseOut(pydantic.BaseModel):
        value: str

    class ExtendedOut(BaseOut):
        extra: int | None = None

    class OldSig(dspy.Signature):
        x: str = dspy.InputField()
        y: BaseOut = dspy.OutputField()

    class NewSig(dspy.Signature):
        x: str = dspy.InputField()
        y: ExtendedOut = dspy.OutputField()

    sot = dspy.StructureOfThought(OldSig, strict=True, temperature=0.3)
    old_sig = sot.signature

    sot.rebind_signature(NewSig)

    assert sot.signature is not old_sig
    assert "y" in sot.signature.output_fields
    assert sot.signature.output_fields["y"].annotation is ExtendedOut

    # Updated predictor should also be a PredictWithStructure with schema in instructions.
    predictor = sot.predictors()[0]
    assert isinstance(predictor, PredictWithStructure)
    schema = json.loads(get_signature(predictor).instructions)
    assert "properties" in schema
    assert "y" in schema["properties"]


def test_predict_with_structure_with_instructions_overrides_schema_used_in_forward_preprocess():
    class SimpleResponse(pydantic.BaseModel):
        answer: str

    class SimpleSignature(dspy.Signature):
        question: str = dspy.InputField()
        response: SimpleResponse = dspy.OutputField()

    pws = PredictWithStructure(SimpleSignature)

    original_sig = get_signature(pws)
    original_schema = json.loads(original_sig.instructions)
    assert "response" in original_schema.get("properties", {})

    # Override instructions with a new JSON schema string.
    new_schema = {
        "type": "object",
        "properties": {
            "response": {"type": "string", "description": "updated schema"},
        },
        "required": ["response"],
    }
    new_schema_str = json.dumps(new_schema, indent=2, sort_keys=True)

    updated_sig = original_sig.with_instructions(new_schema_str)
    # Simulate teleprompter behavior: set the predictor's signature to the updated one.
    pws.signature = updated_sig

    # When PredictWithStructure prepares to call the LM/adapter, it should
    # surface the updated instructions (schema) as the signature it passes along.
    lm = DummyLM([{"response": "dummy"}])
    _, _, sig, _, _ = pws._forward_preprocess(question="What is 1+1?", lm=lm)
    assert sig.instructions == new_schema_str
    assert json.loads(sig.instructions) == new_schema


def test_predict_with_structure_with_schema_adapter_accepts_valid_response():
    class SimpleResponse(pydantic.BaseModel):
        answer: str

    class SimpleSignature(dspy.Signature):
        question: str = dspy.InputField()
        response: SimpleResponse = dspy.OutputField()

    lm = DummyLM([{"response": {"answer": "42"}}])
    dspy.configure(lm=lm)

    pws = PredictWithStructure(SimpleSignature)
    prediction = pws(question="What is 6 * 7?")

    assert isinstance(prediction.response, SimpleResponse)
    assert prediction.response.answer == "42"


def test_predict_with_structure_with_schema_adapter_raises_on_schema_violation():
    class SimpleResponse(pydantic.BaseModel):
        answer: str

    class SimpleSignature(dspy.Signature):
        question: str = dspy.InputField()
        response: SimpleResponse = dspy.OutputField()

    # Start from a normal PredictWithStructure so it computes a schema, then
    # override its instructions with a schema that *also* requires an extra
    # top-level field that the LM will not provide.
    pws = PredictWithStructure(SimpleSignature)
    original_sig = get_signature(pws)

    stricter_schema = {
        "type": "object",
        "properties": {
            "response": {"type": "object"},
            "extra": {"type": "string"},
        },
        "required": ["response", "extra"],
    }
    stricter_schema_str = json.dumps(stricter_schema, indent=2, sort_keys=True)
    updated_sig = original_sig.with_instructions(stricter_schema_str)
    pws.signature = updated_sig

    # LM returns a response that matches the original schema but violates the
    # stricter one (missing the required 'extra' field).
    lm = DummyLM([{"response": {"answer": "42"}}])
    dspy.configure(lm=lm)

    with pytest.raises(AdapterParseError):
        _ = pws(question="What is 6 * 7?")


def test_structure_of_thought_with_schema_adapter_end_to_end():
    class SimpleResponse(pydantic.BaseModel):
        answer: str

    class SimpleSignature(dspy.Signature):
        question: str = dspy.InputField()
        response: SimpleResponse = dspy.OutputField()

    lm = DummyLM([{"response": {"answer": "42"}}])
    dspy.configure(lm=lm)

    sot = dspy.StructureOfThought(SimpleSignature)
    result = sot(question="What is 6 * 7?")

    assert isinstance(result, StructuredResult)
    assert isinstance(result.response, SimpleResponse)
    assert result.response.answer == "42"
