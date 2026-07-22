"""Reproducible before/after benchmark for Syngen text training.

The legacy scenario reproduces the pre-change eager LSTM path. The optimized
scenario measures the new integer-token GRU path with its compiled train step.
Results therefore represent the complete production change, not architecture
performance in isolation.
"""

import argparse
import gc
import json
import string
import time
from statistics import median
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras.models import Model

from syngen.ml.vae.models.features import CharBasedTextFeature


LEGACY_ARCHITECTURE_VERSION = 1
OPTIMIZED_ARCHITECTURE_VERSION = 2


def _legacy_rnn_units(text_max_len: int) -> int:
    if text_max_len < 1:
        return 16
    if text_max_len < 7:
        return 32
    if text_max_len < 13:
        return 128
    if text_max_len < 17:
        return 256
    return 512


def _generate_frame(rows: int, text_max_len: int, columns: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    alphabet = np.asarray(list(string.ascii_uppercase + string.digits))
    data = {}
    for column_index in range(columns):
        lengths = rng.integers(1, text_max_len + 1, size=rows)
        lengths[0] = text_max_len
        data[f"text_{column_index}"] = [
            "".join(rng.choice(alphabet, size=length))
            for length in lengths
        ]
    return pd.DataFrame(data)


def _build_model(
    frame: pd.DataFrame,
    text_max_len: int,
    architecture_version: int,
) -> tuple:
    features = []
    transformed = []
    for name in frame.columns:
        units = (
            _legacy_rnn_units(text_max_len)
            if architecture_version == LEGACY_ARCHITECTURE_VERSION
            else 32
        )
        feature = CharBasedTextFeature(
            name=name,
            text_max_len=text_max_len,
            rnn_units=units,
        )
        feature.architecture_version = architecture_version
        feature.fit(frame[[name]])
        features.append(feature)
        transformed.append(feature.transform(frame[[name]]))

    encoders = [feature.encoder for feature in features]
    encoded = concatenate(encoders) if len(encoders) > 1 else encoders[0]
    global_decoder = Dense(128, activation="relu", name="benchmark_decoder")(encoded)
    outputs = [feature.create_decoder(global_decoder) for feature in features]
    model = Model([feature.input for feature in features], outputs)
    model.add_loss([feature.loss for feature in features])
    return model, transformed


def _train_step(
    model: Model,
    optimizer: tf.keras.optimizers.Optimizer,
    batch: List[tf.Tensor],
) -> tf.Tensor:
    with tf.GradientTape() as tape:
        model(batch, training=True)
        loss = tf.add_n(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def run_scenario(
    architecture: str,
    text_max_len: int,
    columns: int,
    rows: int,
    batch_size: int,
    warmup_steps: int,
    measured_steps: int,
) -> Dict:
    tf.keras.backend.clear_session()
    gc.collect()
    tf.keras.utils.set_random_seed(1729)

    architecture_version = {
        "legacy": LEGACY_ARCHITECTURE_VERSION,
        "optimized": OPTIMIZED_ARCHITECTURE_VERSION,
    }[architecture]
    frame = _generate_frame(rows, text_max_len, columns, seed=1729)
    model, transformed = _build_model(frame, text_max_len, architecture_version)
    batch = [tf.convert_to_tensor(values[:batch_size]) for values in transformed]
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    train_step = (
        tf.function(_train_step, reduce_retracing=True)
        if architecture == "optimized"
        else _train_step
    )

    for _ in range(warmup_steps):
        warmup_loss = train_step(model, optimizer, batch)
        warmup_loss.numpy()

    durations = []
    loss = float("nan")
    for _ in range(measured_steps):
        started = time.perf_counter()
        loss_tensor = train_step(model, optimizer, batch)
        # Materializing the scalar synchronizes accelerator execution so the
        # recorded duration includes model computation rather than dispatch.
        loss = float(loss_tensor.numpy())
        durations.append(time.perf_counter() - started)

    return {
        "architecture": architecture,
        "compiled_train_step": architecture == "optimized",
        "length": text_max_len,
        "columns": columns,
        "rows": rows,
        "batch_size": batch_size,
        "parameters": model.count_params(),
        "transformed_input_bytes": sum(np.asarray(values).nbytes for values in transformed),
        "median_train_step_ms": round(median(durations) * 1000, 3),
        "loss": round(loss, 6),
        "finite_loss": bool(np.isfinite(loss)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=("legacy", "optimized"),
        default=("legacy", "optimized"),
    )
    parser.add_argument("--lengths", nargs="+", type=int, default=(6, 7, 16, 17))
    parser.add_argument("--columns", nargs="+", type=int, default=(1, 2, 4))
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measured-steps", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for architecture in args.architectures:
        for text_max_len in args.lengths:
            for columns in args.columns:
                result = run_scenario(
                    architecture=architecture,
                    text_max_len=text_max_len,
                    columns=columns,
                    rows=args.rows,
                    batch_size=args.batch_size,
                    warmup_steps=args.warmup_steps,
                    measured_steps=args.measured_steps,
                )
                print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
