from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.credit_scorer import CreditScorer
from src.document_extractor import DocumentExtractor
from src.regulatory_rag import RegulatoryRAG
from src.validation import FEATURES


def test_new_document_requires_unit_confirmation_and_displays_validation_error(monkeypatch):
    monkeypatch.setattr(CreditScorer, "load", lambda self: self)
    monkeypatch.setattr(CreditScorer, "predict", lambda self, df: 0.01)
    monkeypatch.setattr(RegulatoryRAG, "load", lambda self: self)
    monkeypatch.setattr(
        pd, "read_csv", lambda *a, **kw: pd.DataFrame([dict.fromkeys(FEATURES, 0.5)])
    )

    def incomplete(self, text):
        raise ValueError("Quick Ratio : donnée manquante")

    monkeypatch.setattr(DocumentExtractor, "extract", incomplete)
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "src/app.py", default_timeout=30
    ).run()
    assert not app.exception
    app.text_area[0].set_value("Rapport incomplet").run()
    audit = next(b for b in app.button if b.label == "Lancer l'Audit Complet")
    assert audit.disabled
    app.checkbox[0].check().run()
    audit = next(b for b in app.button if b.label == "Lancer l'Audit Complet")
    assert not audit.disabled
    audit.click().run()
    assert not app.exception
    assert any("Quick Ratio" in e.value for e in app.error)
