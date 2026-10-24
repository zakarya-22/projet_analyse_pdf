import json

import numpy as np
import pandas as pd
import pytest

from src.credit_scorer import CreditScorer
from src.document_extractor import DocumentExtractor
from src.validation import FEATURES, validate_record


def record():
    return dict.fromkeys(FEATURES, 0.5)


def test_names_normalized_and_zero_preserved():
    data = {k.strip().upper(): v for k, v in record().items()}
    data[FEATURES[0].strip().upper()] = 0
    assert validate_record(data)[FEATURES[0]] == 0


@pytest.mark.parametrize("value", [None, True, "0.5", "12%", float("nan"), float("inf")])
def test_invalid_field_never_imputed(value):
    data = record()
    data[FEATURES[0]] = value
    with pytest.raises(ValueError, match="Operating Profit Rate"):
        validate_record(data)


def test_missing_field_and_duplicate_rejected():
    data = record()
    del data[FEATURES[0]]
    with pytest.raises(ValueError, match="manquante"):
        validate_record(data)
    data = record()
    data[FEATURES[0].strip()] = 0.2
    with pytest.raises(ValueError, match="dupliquée"):
        validate_record(data)


@pytest.mark.parametrize("raw", ["not json", "[]", '{"a": 1}', '{" Operating Profit Rate": null}'])
def test_extractor_invalid_json_or_schema(raw):
    with pytest.raises(ValueError):
        DocumentExtractor.parse(raw)


def test_extractor_valid_json():
    assert DocumentExtractor.parse(json.dumps(record())) == record()


def test_csv_validation_preserves_order_and_identifies_client():
    scorer = CreditScorer("unused", "unused")
    scorer.columns = list(FEATURES)
    df = pd.DataFrame([record()], index=[123])
    df["Client_ID"] = 123
    assert list(scorer._align(df).columns) == list(FEATURES)
    df.loc[123, FEATURES[2]] = np.nan
    with pytest.raises(ValueError, match="Client 123"):
        scorer._align(df)


def test_csv_duplicate_columns_rejected():
    scorer = CreditScorer("unused", "unused")
    scorer.columns = list(FEATURES)
    with pytest.raises(ValueError, match="dupliquées"):
        scorer._align(pd.DataFrame([[1, 2]], columns=[" Quick Ratio", "quick ratio"]))


def test_shap_protective_is_not_a_risk_factor():
    scorer = CreditScorer("unused", "unused")
    scorer.columns = ["debt", "cash", "neutral"]

    class Explainer:
        def shap_values(self, _):
            return np.array([[0.8, -1.2, 0]])

    scorer._explainer = Explainer()
    result = scorer.explain(pd.DataFrame([[1, 2, 3]], columns=scorer.columns))
    assert [f["feature"] for f in result["risk_factors"]] == ["debt"]
    assert [f["feature"] for f in result["protective_factors"]] == ["cash"]
