"""Construccion del pipeline de preprocesamiento."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from vehicular_ml.schema import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def build_preprocessor() -> ColumnTransformer:
    """Crea el preprocesador reproducible de Scikit-learn."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLUMNS),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ]
    )
