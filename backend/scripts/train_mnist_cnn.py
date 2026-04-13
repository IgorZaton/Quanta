from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf


def build_model() -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(16, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Conv2D(32, kernel_size=3, activation="relu"),
            tf.keras.layers.MaxPool2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main() -> None:
    assets_dir = Path(__file__).resolve().parents[1] / "quanta" / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    (x_train, y_train), (x_test, _) = tf.keras.datasets.mnist.load_data()
    x_train = (x_train.astype(np.float32) / 255.0)[..., np.newaxis]
    x_test = (x_test.astype(np.float32) / 255.0)[..., np.newaxis]

    model = build_model()
    model.fit(x_train, y_train, epochs=2, batch_size=128, validation_split=0.1, verbose=2)

    model_path = assets_dir / "mnist_cnn.keras"
    dataset_path = assets_dir / "mnist_test.npy"
    model.save(model_path)

    # Keep test dataset small so package stays lightweight.
    np.save(dataset_path, x_test[:512].astype(np.float32))
    print(f"Saved model: {model_path}")
    print(f"Saved dataset: {dataset_path}")


if __name__ == "__main__":
    main()
