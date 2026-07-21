"""Red neuronal pequeña y opcional para evitar imponer TensorFlow al flujo básico."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quinielator.models.base import BatchPrediction, PredictionModel, score_probabilities


class KerasMatchModel(PredictionModel):
    """MLP regularizado que estima dos intensidades de gol positivas."""

    name = "keras"

    def __init__(
        self,
        max_score: int = 8,
        seed: int = 42,
        epochs: int = 80,
        batch_size: int = 32,
        patience: int = 10,
    ) -> None:
        super().__init__(max_score, seed)
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.preprocessor = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                ),
                ("scaler", StandardScaler()),
            ]
        )
        self.model: Any | None = None
        self.history: dict[str, list[float]] = {}

    @staticmethod
    def available() -> bool:
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            return False
        return True

    def _build(self, input_size: int) -> Any:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(self.seed)
        inputs = tf.keras.Input(shape=(input_size,), name="features")
        hidden = tf.keras.layers.Dense(32, activation="relu")(inputs)
        hidden = tf.keras.layers.Dropout(0.20)(hidden)
        hidden = tf.keras.layers.Dense(16, activation="relu")(hidden)
        outputs = tf.keras.layers.Dense(2, activation="softplus", name="expected_goals")(hidden)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name="quinielator_goals")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
            loss=tf.keras.losses.Poisson(),
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
        )
        return model

    def fit(self, features: pd.DataFrame, targets: pd.DataFrame) -> KerasMatchModel:
        if not self.available():
            raise RuntimeError('TensorFlow no está instalado. Usa: pip install -e ".[keras]"')
        import tensorflow as tf

        transformed = self.preprocessor.fit_transform(features).astype(np.float32)
        goal_targets = targets[["home_goals", "away_goals"]].to_numpy(dtype=np.float32)
        self.model = self._build(transformed.shape[1])
        validation_size = max(1, int(len(transformed) * 0.15))
        split = max(1, len(transformed) - validation_size)
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=self.patience,
            restore_best_weights=True,
        )
        result = self.model.fit(
            transformed[:split],
            goal_targets[:split],
            validation_data=(transformed[split:], goal_targets[split:]),
            epochs=self.epochs,
            batch_size=self.batch_size,
            shuffle=False,
            verbose=0,
            callbacks=[callback],
        )
        self.history = {
            key: [float(value) for value in values] for key, values in result.history.items()
        }
        return self

    def predict(self, features: pd.DataFrame) -> BatchPrediction:
        if self.model is None:
            raise RuntimeError("El modelo Keras todavía no fue entrenado")
        transformed = self.preprocessor.transform(features).astype(np.float32)
        goals = np.asarray(self.model.predict(transformed, verbose=0), dtype=float)
        return score_probabilities(
            np.clip(goals[:, 0], 0.05, 5.0),
            np.clip(goals[:, 1], 0.05, 5.0),
            self.max_score,
        )

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("No se puede guardar un modelo sin entrenar")
        path.mkdir(parents=True, exist_ok=True)
        self.model.save(path / "model.keras")
        joblib.dump(self.preprocessor, path / "preprocessor.joblib")
        joblib.dump(
            {
                "max_score": self.max_score,
                "seed": self.seed,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "patience": self.patience,
                "history": self.history,
            },
            path / "metadata.joblib",
        )

    @classmethod
    def load_from(cls, path: Path) -> KerasMatchModel:
        """Restaura red, preprocesador y configuración desde un directorio."""

        if not cls.available():
            raise RuntimeError("TensorFlow es necesario para cargar este modelo")
        import tensorflow as tf

        metadata = joblib.load(path / "metadata.joblib")
        instance = cls(
            max_score=int(metadata["max_score"]),
            seed=int(metadata["seed"]),
            epochs=int(metadata["epochs"]),
            batch_size=int(metadata["batch_size"]),
            patience=int(metadata["patience"]),
        )
        instance.preprocessor = joblib.load(path / "preprocessor.joblib")
        instance.model = tf.keras.models.load_model(path / "model.keras")
        instance.history = metadata.get("history", {})
        return instance
