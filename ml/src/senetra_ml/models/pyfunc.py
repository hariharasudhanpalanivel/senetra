"""MLflow pyfunc wrapper so bundles are versioned, registered and served through the registry."""

from __future__ import annotations

import warnings
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature

from senetra_ml.models.bundle import ForecastBundle

DEFAULT_PARAMS = {"level": "phc", "interval_kind": "scenario"}
PIP_REQUIREMENTS = ["numpy>=2.0", "pandas>=2.2", "mlflow>=3.1,<4"]

# The model validates its own input columns and declares an explicit signature instead of type hints.
warnings.filterwarnings("ignore", message=".*Add type hints to the `predict` method.*")
# Id and horizon columns are integers by contract and never missing.
warnings.filterwarnings("ignore", message=".*Inferred schema contains integer column.*")


class SenetraForecastModel(mlflow.pyfunc.PythonModel):
    """Input: one row per (PHC, horizon) with ids, dates and raw features.

    Params: level (phc|district|country|world) and interval_kind (scenario|observed).
    """

    def load_context(self, context) -> None:
        self.bundle = ForecastBundle.load(context.artifacts["bundle"])

    def predict(self, context, model_input, params=None):
        params = {**DEFAULT_PARAMS, **(params or {})}
        return self.bundle.forecast(pd.DataFrame(model_input), level=params["level"],
                                    interval_kind=params["interval_kind"])


def log_forecast_model(bundle: ForecastBundle, bundle_dir: Path, input_example: pd.DataFrame,
                       registered_model_name: str | None):
    output_example = bundle.forecast(input_example)
    signature = infer_signature(input_example, output_example, params=DEFAULT_PARAMS)
    return mlflow.pyfunc.log_model(
        name="model",
        python_model=SenetraForecastModel(),
        artifacts={"bundle": str(bundle_dir)},
        signature=signature,
        input_example=input_example,
        pip_requirements=PIP_REQUIREMENTS,
        registered_model_name=registered_model_name,
    )
