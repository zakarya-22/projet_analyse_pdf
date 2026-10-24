"""Shared contract for extracted records and CSV scoring inputs."""

import math
from numbers import Real

FEATURES = (
    " Operating Profit Rate",
    " Research and development expense rate",
    " Interest-bearing debt interest rate",
    " Tax rate (A)",
    " After-tax Net Profit Growth Rate",
    " Net Value Growth Rate",
    " Quick Ratio",
    " Borrowing dependency",
    " Total Asset Turnover",
    " Revenue per person",
    " Allocation rate per person",
    " Quick Assets/Total Assets",
    " Cash/Total Assets",
    " Current Liabilities/Liability",
    " Liability-Assets Flag",
)


def validate_record(data, columns=FEATURES, *, allow_extra=False):
    """Normalize names; never impute or guess training units/ranges."""
    if not isinstance(data, dict):
        raise ValueError("Les indicateurs doivent former un objet JSON.")
    normalized = {}
    for key, value in data.items():
        name = str(key).strip().casefold()
        if name in normalized:
            raise ValueError(f"Colonne dupliquée après normalisation : {key}")
        normalized[name] = value
    expected = {str(c).strip().casefold() for c in columns}
    extras = normalized.keys() - expected
    if extras and not allow_extra:
        raise ValueError(f"Indicateurs inconnus : {', '.join(sorted(extras))}")
    result, errors = {}, []
    for column in columns:
        value = normalized.get(str(column).strip().casefold())
        if value is None:
            errors.append(f"{column.strip()} : donnée manquante")
        elif isinstance(value, bool) or not isinstance(value, Real):
            errors.append(f"{column.strip()} : nombre attendu (unité explicite requise)")
        elif not math.isfinite(float(value)):
            errors.append(f"{column.strip()} : nombre fini attendu")
        else:
            result[column] = float(value)
    if errors:
        raise ValueError("Dossier incomplet ou invalide. " + "; ".join(errors))
    return result
