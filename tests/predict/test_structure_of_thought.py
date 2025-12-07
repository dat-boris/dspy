import json

import pytest

import dspy
import pydantic
from dspy.predict.structure_of_thought import StructuredResult, _validate_reasoning_superset


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

    # Should be valid JSON and pretty-printed (multi-line with indentation).
    parsed = json.loads(schema_text)
    assert isinstance(parsed, dict)
    assert "{\n" in schema_text

