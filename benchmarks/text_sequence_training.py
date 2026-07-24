"""Reproducible before/after benchmark for Syngen text training.

The legacy scenario reproduces the pre-change eager LSTM path. The optimized
scenario measures the new integer-token GRU path with its compiled train step.
Results therefore represent the complete production change, not architecture
performance in isolation.

Two comparisons are supported:

* ``--mode perf`` (default): median train-step latency, parameter count, and
  transformed-input size, matching the original perf-only benchmark.
* ``--mode quality``: an architecture-independent per-character accuracy and
  loss trend over several epochs. This exists because the two architectures'
  raw training losses are NOT directly comparable -- the legacy loss omits
  ``+1`` from ``vocab_size``, so the rarest character per column (the one
  Keras's ``Tokenizer`` assigns the highest id) falls outside the one-hot
  depth and silently contributes zero loss no matter what is predicted for
  it. ``run_quality_scenario`` sidesteps that by comparing ``argmax`` decoder
  predictions against the true integer token ids (independent of vocab_size
  and of the one-hot encoding), so legacy's blind spot shows up as reduced
  accuracy instead of being hidden inside a lower loss number.

``--pattern`` selects the synthetic data used by ``--mode quality``:

* ``random`` (default): uniformly random alphanumeric strings of varying
  length, matching the original perf-fixture data. Useful for confirming a
  capacity change does not regress unstructured/near-random text columns.
* ``structured``: a deterministic literal prefix (covering ``--prefix-ratio``
  of ``text_max_len``) followed by a random suffix, e.g. a stand-in for
  templated columns such as ``https://www.<domain>/<slug>-<digits>``. This is
  what surfaced a long-structured-column fidelity regression under the first
  optimized architecture (a flat ``rnn_units=32`` GRU regardless of length):
  the model's confidence on the deterministic prefix collapsed well below the
  legacy length-scaled LSTM's, so the existing ``top_p`` nucleus sampling in
  ``CharBasedTextFeature.inverse_transform`` corrupted characters that were
  supposed to be constant across every row. Production now records token
  positions that are invariant across every training row and restores those
  exact tokens after stochastic sampling. This keeps the fast fixed-width
  GRU while guaranteeing fixed prefixes such as ``https://www.`` remain
  intact -- see tmp/22-07-26/syngen-text-lstm-perf-findings.md for the full
  derivation and benchmark evidence. ``optimized`` in this module always
  means "whatever architecture_version syngen.ml.vae.models.features assigns
  to newly created features" (currently 3), so this benchmark keeps tracking
  production behavior as it evolves.

``--units-override`` lets ``--mode perf``/``--mode quality`` force a specific
fixed GRU width for the optimized architecture (bypassing the production
sizing policy), so a range of candidate widths can be swept against both
patterns before choosing/revisiting that policy.
"""

import argparse
import gc
import json
import string
import time
from statistics import median
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, concatenate
from tensorflow.keras.models import Model

import syngen.ml.vae.models.features as features_module
from syngen.ml.vae.models.features import CharBasedTextFeature


LEGACY_ARCHITECTURE_VERSION = features_module.LEGACY_TEXT_ARCHITECTURE_VERSION
# Always tracks whatever architecture_version production assigns to newly
# created features (currently VARIABLE_WIDTH_TEXT_ARCHITECTURE_VERSION), so
# "optimized" in this benchmark never silently drifts from what a fresh
# `fit()` actually does in production.
OPTIMIZED_ARCHITECTURE_VERSION = features_module.TEXT_ARCHITECTURE_VERSION

# Captured once at import time so units_override can always be restored to
# the real production default afterwards, regardless of how many times
# _build_model has overridden the module-level constant in this process.
_DEFAULT_TEXT_RNN_UNITS = features_module.TEXT_RNN_UNITS

# A fixed, deterministic literal used to build the "structured" pattern's
# constant prefix. Long enough to cover the longest --prefix-ratio *
# --lengths combination exercised by the sweep; sliced, never repeated, so
# the prefix is unambiguous character-by-character (no periodicity that
# could make the task artificially easier).
_STRUCTURED_PREFIX_LITERAL = "https://www.template-prefix-marker.example/path/"


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


def structured_prefix_length(text_max_len: int, prefix_ratio: float) -> int:
    """Length of the deterministic literal prefix used by ``_generate_structured_frame``.

    Exposed so callers can score prefix fidelity without duplicating the
    ratio-to-length arithmetic used at generation time.
    """
    return max(1, min(text_max_len, round(text_max_len * prefix_ratio)))


def _generate_structured_frame(
    rows: int,
    text_max_len: int,
    columns: int,
    seed: int,
    prefix_ratio: float = 0.35,
) -> pd.DataFrame:
    """Deterministic literal prefix + random suffix, e.g. a URL-template stand-in.

    Every row in a column shares the exact same prefix
    (``structured_prefix_length(text_max_len, prefix_ratio)`` characters); only
    the suffix varies. This isolates whether an architecture can reliably
    reconstruct a long constant substring, independent of the (harder,
    genuinely unlearnable) variable content around it.
    """
    prefix_len = structured_prefix_length(text_max_len, prefix_ratio)
    if prefix_len > len(_STRUCTURED_PREFIX_LITERAL):
        raise ValueError(
            "text_max_len * prefix_ratio exceeds the built-in literal length; "
            "lower prefix_ratio or extend _STRUCTURED_PREFIX_LITERAL."
        )
    prefix = _STRUCTURED_PREFIX_LITERAL[:prefix_len]
    suffix_len = text_max_len - prefix_len

    rng = np.random.default_rng(seed)
    alphabet = np.asarray(list(string.ascii_lowercase + string.digits))
    data = {}
    for column_index in range(columns):
        data[f"text_{column_index}"] = [
            prefix + "".join(rng.choice(alphabet, size=suffix_len))
            for _ in range(rows)
        ]
    return pd.DataFrame(data)


def _build_model(
    frame: pd.DataFrame,
    text_max_len: int,
    architecture_version: int,
    units_override: Optional[int] = None,
) -> tuple:
    features = []
    transformed = []
    for name in frame.columns:
        if architecture_version == LEGACY_ARCHITECTURE_VERSION:
            units = _legacy_rnn_units(text_max_len)
        else:
            units = units_override if units_override is not None else _DEFAULT_TEXT_RNN_UNITS
        feature = CharBasedTextFeature(
            name=name,
            text_max_len=text_max_len,
            rnn_units=units,
        )
        feature.architecture_version = architecture_version
        if architecture_version != LEGACY_ARCHITECTURE_VERSION:
            # CharBasedTextFeature._text_rnn_units reads the module-level
            # TEXT_RNN_UNITS constant for every optimized architecture.
            # Restoring it when no override is supplied prevents one sweep
            # scenario from leaking into the next in the same process.
            features_module.TEXT_RNN_UNITS = (
                units_override if units_override is not None else _DEFAULT_TEXT_RNN_UNITS
            )
        feature.fit(frame[[name]])
        features.append(feature)
        transformed.append(feature.transform(frame[[name]]))

    encoders = [feature.encoder for feature in features]
    encoded = concatenate(encoders) if len(encoders) > 1 else encoders[0]
    global_decoder = Dense(128, activation="relu", name="benchmark_decoder")(encoded)
    outputs = [feature.create_decoder(global_decoder) for feature in features]
    model = Model([feature.input for feature in features], outputs)
    model.add_loss([feature.loss for feature in features])
    if not features:
        units_used = None
    elif architecture_version == LEGACY_ARCHITECTURE_VERSION:
        units_used = _legacy_rnn_units(text_max_len)
    else:
        units_used = features[0]._text_rnn_units
    return model, transformed, features, units_used


def _true_token_ids(feature: CharBasedTextFeature, column: pd.Series) -> np.ndarray:
    """Architecture-independent ground truth token ids for a text column.

    Mirrors ``CharBasedTextFeature.transform`` up to (but not including) the
    legacy one-hot / optimized int32 branch, so the same integer ids are used
    as the accuracy label regardless of ``architecture_version``. Unlike the
    training loss, this label set is not affected by the +1 vocab_size gap
    between the two architectures.
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    sequences = feature.tokenizer.texts_to_sequences(column)
    return pad_sequences(
        sequences,
        maxlen=feature.text_max_len,
        padding="post",
        truncating="post",
        value=0.0,
    ).astype("int32")


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
    units_override: Optional[int] = None,
) -> Dict:
    tf.keras.backend.clear_session()
    gc.collect()
    tf.keras.utils.set_random_seed(1729)

    architecture_version = {
        "legacy": LEGACY_ARCHITECTURE_VERSION,
        "optimized": OPTIMIZED_ARCHITECTURE_VERSION,
    }[architecture]
    frame = _generate_frame(rows, text_max_len, columns, seed=1729)
    model, transformed, _features, units_used = _build_model(
        frame, text_max_len, architecture_version, units_override=units_override
    )
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
        "rnn_units": units_used,
        "parameters": model.count_params(),
        "transformed_input_bytes": sum(np.asarray(values).nbytes for values in transformed),
        "median_train_step_ms": round(median(durations) * 1000, 3),
        "loss": round(loss, 6),
        "finite_loss": bool(np.isfinite(loss)),
    }


def run_quality_scenario(
    architecture: str,
    text_max_len: int,
    columns: int,
    rows: int,
    batch_size: int,
    epochs: int,
    report_every: int,
    pattern: str = "random",
    units_override: Optional[int] = None,
    prefix_ratio: float = 0.35,
) -> Dict:
    """Train for several epochs and report a fair, vocab-size-independent metric.

    Both the loss trend (comparable only within one architecture, not across
    them -- see the module docstring) and a character accuracy trend
    (comparable across architectures) are returned so the two can be judged
    side by side instead of relying on the misleading raw loss numbers.

    With ``pattern="structured"``, three additional keys are returned, all
    scoring only the deterministic literal prefix (see
    ``_generate_structured_frame``):

    * ``final_prefix_accuracy``: argmax-vs-true-id match rate.
    * ``final_prefix_confidence``: mean softmax probability mass on the
      correct id.
    * ``final_prefix_generation_match_rate``: exact-prefix match rate using
      the real ``CharBasedTextFeature.inverse_transform`` stochastic
      ``top_p`` sampling path -- i.e. what a user actually sees when
      generating data. Confidence can be well below 1.0 while accuracy is
      still 100% (argmax still picks the right character), and a lower
      confidence directly translates into more sampling-induced corruption
      of what should be a constant substring. This is the metric that
      exposed the long-structured-column regression.
    """
    if report_every < 1:
        raise ValueError("report_every must be a positive integer")
    if pattern not in ("random", "structured"):
        raise ValueError("pattern must be 'random' or 'structured'")

    tf.keras.backend.clear_session()
    gc.collect()
    tf.keras.utils.set_random_seed(1729)

    architecture_version = {
        "legacy": LEGACY_ARCHITECTURE_VERSION,
        "optimized": OPTIMIZED_ARCHITECTURE_VERSION,
    }[architecture]
    if pattern == "structured":
        frame = _generate_structured_frame(
            rows, text_max_len, columns, seed=1729, prefix_ratio=prefix_ratio
        )
        prefix_len = structured_prefix_length(text_max_len, prefix_ratio)
    else:
        frame = _generate_frame(rows, text_max_len, columns, seed=1729)
        prefix_len = None
    model, transformed, features, units_used = _build_model(
        frame, text_max_len, architecture_version, units_override=units_override
    )
    true_ids = tuple(
        _true_token_ids(feature, frame[name])
        for feature, name in zip(features, frame.columns)
    )

    dataset = tf.data.Dataset.from_tensor_slices((tuple(transformed), true_ids))
    dataset = dataset.shuffle(rows, seed=1729, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    train_step = (
        tf.function(_train_step, reduce_retracing=True)
        if architecture == "optimized"
        else _train_step
    )

    loss_trace = []
    accuracy_trace = []
    for _ in range(epochs):
        epoch_loss_total = 0.0
        epoch_batches = 0
        epoch_correct = 0.0
        epoch_count = 0
        for batch_inputs, batch_true_ids in dataset:
            batch_inputs = list(batch_inputs)
            loss_tensor = train_step(model, optimizer, batch_inputs)
            epoch_loss_total += float(loss_tensor.numpy())
            epoch_batches += 1

            outputs = tf.nest.flatten(model(batch_inputs, training=False))
            for logits, target in zip(outputs, batch_true_ids):
                predicted = tf.argmax(logits, axis=-1, output_type=tf.int32)
                matches = tf.cast(tf.equal(predicted, target), tf.float32)
                epoch_correct += float(tf.reduce_sum(matches).numpy())
                epoch_count += int(tf.size(matches).numpy())

        loss_trace.append(round(epoch_loss_total / epoch_batches, 6))
        accuracy_trace.append(round(epoch_correct / epoch_count, 6))

    def _sample(trace: List[float]) -> List[float]:
        sampled = trace[::report_every]
        if sampled[-1] != trace[-1]:
            sampled.append(trace[-1])
        return sampled

    result = {
        "architecture": architecture,
        "pattern": pattern,
        "length": text_max_len,
        "columns": columns,
        "rows": rows,
        "batch_size": batch_size,
        "epochs": epochs,
        "rnn_units": units_used,
        "initial_loss": loss_trace[0],
        "final_loss": loss_trace[-1],
        "initial_character_accuracy": accuracy_trace[0],
        "final_character_accuracy": accuracy_trace[-1],
        "loss_trace": _sample(loss_trace),
        "character_accuracy_trace": _sample(accuracy_trace),
    }

    if pattern == "structured":
        outputs = tf.nest.flatten(model([tf.convert_to_tensor(v) for v in transformed], training=False))
        prefix_correct = 0.0
        prefix_confidence_total = 0.0
        prefix_count = 0
        generation_match = 0.0
        generation_count = 0
        expected_prefix = frame[frame.columns[0]].iloc[0][:prefix_len]
        for feature, logits, target in zip(features, outputs, true_ids):
            probs = tf.nn.softmax(logits, axis=-1).numpy()
            predicted = np.argmax(logits.numpy(), axis=-1)
            prefix_match = predicted[:, :prefix_len] == target[:, :prefix_len]
            prefix_confidence = probs[:, :prefix_len, :].max(axis=-1)
            prefix_correct += float(prefix_match.sum())
            prefix_confidence_total += float(prefix_confidence.sum())
            prefix_count += prefix_match.size

            # This is the metric that matches what a user actually observes:
            # CharBasedTextFeature.inverse_transform uses stochastic top_p=0.9
            # nucleus sampling, not argmax, so a high argmax accuracy does not
            # by itself guarantee the deterministic prefix renders correctly
            # every time it is generated.
            generated = feature.inverse_transform(logits.numpy())
            generation_match += sum(
                1 for text in generated if text[:prefix_len] == expected_prefix
            )
            generation_count += len(generated)

        result["prefix_length"] = prefix_len
        result["final_prefix_accuracy"] = round(prefix_correct / prefix_count, 6)
        result["final_prefix_confidence"] = round(prefix_confidence_total / prefix_count, 6)
        result["final_prefix_generation_match_rate"] = round(
            generation_match / generation_count, 6
        )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("perf", "quality"),
        default="perf",
        help=(
            "'perf' measures train-step latency/params (default, backward "
            "compatible). 'quality' measures a fair, architecture-independent "
            "loss/accuracy trend over several epochs."
        ),
    )
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
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="Number of training epochs for --mode quality.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=5,
        help="Sampling stride for the loss/accuracy traces in --mode quality.",
    )
    parser.add_argument(
        "--pattern",
        choices=("random", "structured"),
        default="random",
        help=(
            "Synthetic data shape for --mode quality: 'random' (default) or "
            "'structured' (deterministic literal prefix + random suffix, "
            "e.g. a URL-template stand-in)."
        ),
    )
    parser.add_argument(
        "--prefix-ratio",
        type=float,
        default=0.35,
        help="Fraction of text_max_len that is the deterministic literal prefix "
        "for --pattern structured.",
    )
    parser.add_argument(
        "--units-override",
        type=int,
        default=None,
        help="Force a specific fixed GRU width for the optimized architecture, "
        "bypassing the production TEXT_RNN_UNITS constant. For sweeping "
        "candidate widths only; has no effect on the legacy architecture.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for architecture in args.architectures:
        for text_max_len in args.lengths:
            for columns in args.columns:
                if args.mode == "quality":
                    result = run_quality_scenario(
                        architecture=architecture,
                        text_max_len=text_max_len,
                        columns=columns,
                        rows=args.rows,
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        report_every=args.report_every,
                        pattern=args.pattern,
                        units_override=args.units_override,
                        prefix_ratio=args.prefix_ratio,
                    )
                else:
                    result = run_scenario(
                        architecture=architecture,
                        text_max_len=text_max_len,
                        columns=columns,
                        rows=args.rows,
                        batch_size=args.batch_size,
                        warmup_steps=args.warmup_steps,
                        measured_steps=args.measured_steps,
                        units_override=args.units_override,
                    )
                print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
