"""
Framework compatibility utilities for tests.

This module provides adapters that normalize the API between TensorFlow (Keras)
and PyTorch, so that test BODIES can stay unchanged during migration.

TO MIGRATE FROM TF TO PYTORCH:
  Update ONLY the functions in this file — do not touch individual test files.
"""

import numpy as np
from typing import List

# ---------------------------------------------------------------------------
# Tensor / array utilities
# ---------------------------------------------------------------------------

def to_numpy(x) -> np.ndarray:
    """
    Convert any framework tensor, or plain array-like, to a numpy array.

    TF:      EagerTensor  → .numpy()
    PyTorch: Tensor       → .detach().cpu().numpy()
    Other:   np.asarray() fallback
    """
    if x is None:
        return None
    if hasattr(x, "detach"):          # PyTorch Tensor
        return x.detach().cpu().numpy()
    if hasattr(x, "numpy"):           # TF EagerTensor
        return x.numpy()
    return np.asarray(x)


def to_float(x) -> float:
    """Return x as a plain Python float (works for TF/PT scalars and numpy)."""
    arr = to_numpy(x)
    return float(np.squeeze(arr))


def make_zeros(shape, dtype=np.float32) -> np.ndarray:
    """Return a zeros numpy array of the given shape."""
    return np.zeros(shape, dtype=dtype)


def make_randn(shape, dtype=np.float32) -> np.ndarray:
    """Return a standard-normal numpy array of the given shape."""
    return np.random.randn(*shape).astype(dtype)


# ---------------------------------------------------------------------------
# Model / layer introspection
# ---------------------------------------------------------------------------

def is_trainable_model(obj) -> bool:
    """
    Return True if *obj* is a trainable model in either framework.

    TF/Keras:  model.trainable_weights
    PyTorch:   model.parameters()
    """
    return hasattr(obj, "trainable_weights") or hasattr(obj, "parameters")


def get_param_count(model) -> int:
    """Return total number of trainable scalar parameters."""
    if hasattr(model, "trainable_weights"):           # Keras
        return int(sum(np.prod(w.shape) for w in model.trainable_weights))
    if hasattr(model, "parameters"):                  # PyTorch
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return 0


def get_param_arrays(model) -> List[np.ndarray]:
    """Return all trainable parameters as a list of numpy arrays."""
    if hasattr(model, "trainable_weights"):
        return [to_numpy(w) for w in model.trainable_weights]
    if hasattr(model, "parameters"):
        return [to_numpy(p) for p in model.parameters() if p.requires_grad]
    return []


# ---------------------------------------------------------------------------
# Layer forward-pass adapters
# (Change ONLY these blocks when migrating to PyTorch)
# ---------------------------------------------------------------------------

def fwd_feature_loss_layer(layer, feature_input: np.ndarray,
                           feature_decoder: np.ndarray) -> np.ndarray:
    """
    Execute FeatureLossLayer forward pass and return the decoder output.

    TF path  : wraps in a minimal Keras functional model (required for add_loss).
    PT path  : calls layer(input, decoder) directly; handles (output, loss) tuple.
    """
    # ---- TF path ----
    try:
        import tensorflow as tf
        if isinstance(layer, tf.keras.layers.Layer):
            inp_a = tf.keras.Input(shape=feature_input.shape[1:])
            inp_b = tf.keras.Input(shape=feature_decoder.shape[1:])
            out = layer([inp_a, inp_b])
            model = tf.keras.Model(inputs=[inp_a, inp_b], outputs=out)
            return to_numpy(model([
                tf.constant(feature_input),
                tf.constant(feature_decoder),
            ]))
    except (ImportError, AttributeError):
        pass

    # ---- PyTorch path ----
    import torch
    with torch.no_grad():
        result = layer(
            torch.as_tensor(feature_input),
            torch.as_tensor(feature_decoder),
        )
    if isinstance(result, (tuple, list)):
        return to_numpy(result[0])
    return to_numpy(result)


def fwd_feature_loss_layer_with_losses(layer, feature_input: np.ndarray,
                                       feature_decoder: np.ndarray):
    """
    Execute FeatureLossLayer and return (decoder_output, [loss_values]).

    TF path  : runs via Keras model; losses taken from model.losses.
    PT path  : loss returned as second element of tuple.
    """
    # ---- TF path ----
    try:
        import tensorflow as tf
        if isinstance(layer, tf.keras.layers.Layer):
            inp_a = tf.keras.Input(shape=feature_input.shape[1:])
            inp_b = tf.keras.Input(shape=feature_decoder.shape[1:])
            out = layer([inp_a, inp_b])
            model = tf.keras.Model(inputs=[inp_a, inp_b], outputs=out)
            result = to_numpy(model([
                tf.constant(feature_input),
                tf.constant(feature_decoder),
            ]))
            losses = [to_float(l) for l in model.losses]
            return result, losses
    except (ImportError, AttributeError):
        pass

    # ---- PyTorch path ----
    import torch
    with torch.no_grad():
        result = layer(
            torch.as_tensor(feature_input),
            torch.as_tensor(feature_decoder),
        )
    if isinstance(result, (tuple, list)):
        return to_numpy(result[0]), [to_float(result[1])]
    return to_numpy(result), []


def fwd_sample_layer(layer, mean: np.ndarray, log_var: np.ndarray) -> np.ndarray:
    """
    Execute SampleLayer forward pass and return the sampled z.

    TF path  : wraps in a Keras model (add_loss requires symbolic context).
    PT path  : calls layer(mean, log_var) directly; handles (z, kl) tuple.
    """
    # ---- TF path ----
    try:
        import tensorflow as tf
        if isinstance(layer, tf.keras.layers.Layer):
            latent_dim = mean.shape[-1]
            m_inp = tf.keras.Input(shape=(latent_dim,))
            lv_inp = tf.keras.Input(shape=(latent_dim,))
            output = layer([m_inp, lv_inp])
            model = tf.keras.Model(inputs=[m_inp, lv_inp], outputs=output)
            return to_numpy(model([
                tf.constant(mean),
                tf.constant(log_var),
            ]))
    except (ImportError, AttributeError):
        pass

    # ---- PyTorch path ----
    import torch
    with torch.no_grad():
        result = layer(
            torch.as_tensor(mean, dtype=torch.float32),
            torch.as_tensor(log_var, dtype=torch.float32),
        )
    if isinstance(result, (tuple, list)):
        return to_numpy(result[0])
    return to_numpy(result)


# ---------------------------------------------------------------------------
# CVAE / model adapters
# ---------------------------------------------------------------------------

def build_or_init_cvae(cvae) -> None:
    """
    Make a CVAE 'ready to use' after construction.

    TF  : calls cvae.build_model()
    PT  : model already built in __init__; no-op or calls build() if present
    """
    if hasattr(cvae, "build_model"):
        cvae.build_model()                              # Keras / current TF path
    # PyTorch: construction already creates all nn.Module attributes


def is_cvae_built(cvae) -> bool:
    """Return True if the CVAE has a built main model."""
    model = getattr(cvae, "model", None)
    return model is not None and is_trainable_model(model)


def run_encoder(cvae, inputs_list: List[np.ndarray]) -> np.ndarray:
    """
    Run the encoder sub-model and return the latent representation.

    TF  : cvae.encoder_model.predict(inputs_list)
    PT  : cvae.encode(tensors) or cvae.encoder_model(tensors)
    """
    # ---- TF path ----
    try:
        import tensorflow as tf
        if hasattr(cvae, "encoder_model") and isinstance(
            cvae.encoder_model, tf.keras.Model
        ):
            tf_inputs = [tf.constant(x) for x in inputs_list]
            return to_numpy(cvae.encoder_model(tf_inputs))
    except (ImportError, AttributeError):
        pass

    # ---- PyTorch path ----
    import torch
    with torch.no_grad():
        pt_inputs = [torch.as_tensor(x) for x in inputs_list]
        if hasattr(cvae, "encoder_model"):
            return to_numpy(cvae.encoder_model(pt_inputs))
        return to_numpy(cvae.encode(pt_inputs))


def run_generator(cvae, latent: np.ndarray) -> List[np.ndarray]:
    """
    Run the generator sub-model given a latent vector.

    Returns: list of numpy arrays, one per feature.
    """
    # ---- TF path ----
    try:
        import tensorflow as tf
        if hasattr(cvae, "generator_model") and isinstance(
            cvae.generator_model, tf.keras.Model
        ):
            result = cvae.generator_model(tf.constant(latent))
            if isinstance(result, (list, tuple)):
                return [to_numpy(r) for r in result]
            return [to_numpy(result)]
    except (ImportError, AttributeError):
        pass

    # ---- PyTorch path ----
    import torch
    with torch.no_grad():
        result = cvae.generator_model(torch.as_tensor(latent))
    if isinstance(result, (list, tuple)):
        return [to_numpy(r) for r in result]
    return [to_numpy(result)]


def get_model_num_inputs(cvae) -> int:
    """Return number of inputs (features) the main model expects."""
    if hasattr(cvae, "model"):
        m = cvae.model
        if hasattr(m, "inputs"):       # Keras
            return len(m.inputs)
        if hasattr(m, "input_keys"):   # custom PyTorch dict-model
            return len(m.input_keys)
    return len(cvae.inputs)            # fallback: list built in build_model


def get_generator_num_outputs(cvae) -> int:
    """Return number of outputs the generator model produces."""
    if hasattr(cvae, "generator_model"):
        m = cvae.generator_model
        if hasattr(m, "outputs"):      # Keras
            return len(m.outputs)
    # PyTorch fallback: run a dummy forward
    latent = make_zeros((1, cvae.latent_dim))
    outputs = run_generator(cvae, latent)
    return len(outputs)


def get_generator_input_dim(cvae) -> int:
    """Return the expected input dimensionality of the generator."""
    if hasattr(cvae, "generator_model"):
        m = cvae.generator_model
        if hasattr(m, "input_shape"):   # Keras
            return int(m.input_shape[-1])
    return cvae.latent_dim              # fallback


def get_encoder_output_dim(cvae) -> int:
    """Return output dimensionality of the encoder (= latent_dim for mu)."""
    if hasattr(cvae, "encoder_model"):
        m = cvae.encoder_model
        if hasattr(m, "output_shape"):  # Keras
            return int(m.output_shape[-1])
    # PyTorch fallback: run dummy
    n_features = len(cvae.dataset.features)
    dummy_inputs = [
        make_zeros((1, f.input_dimension))
        for f in cvae.dataset.features.values()
    ]
    encoded = run_encoder(cvae, dummy_inputs)
    return encoded.shape[-1]


# ---------------------------------------------------------------------------
# Optimizer / loss-metric adapters
# ---------------------------------------------------------------------------

def get_optimizer_lr(optimizer) -> float:
    """Return the current learning rate of an optimizer (TF or PyTorch)."""
    if hasattr(optimizer, "learning_rate"):          # Keras optimizer
        return float(optimizer.learning_rate)
    if hasattr(optimizer, "param_groups"):           # PyTorch optimizer
        return float(optimizer.param_groups[0]["lr"])
    raise AttributeError(f"Cannot determine LR from {type(optimizer)}")


def accumulate_and_get_mean(loss_metric, values) -> float:
    """
    Feed values into a loss accumulator and return the mean.

    TF  : tf.keras.metrics.Mean — uses update_state (called via __call__)
    PT  : simple running mean dict or custom class
    """
    # ---- TF path ----
    try:
        import tensorflow as tf
        if isinstance(loss_metric, tf.keras.metrics.Metric):
            loss_metric.reset_states()
            for v in values:
                loss_metric(tf.constant(float(v)))
            return float(loss_metric.result().numpy())
    except (ImportError, AttributeError):
        pass

    # ---- PyTorch / plain path ----
    if hasattr(loss_metric, "reset"):
        loss_metric.reset()
    total, count = 0.0, 0
    for v in values:
        total += float(v)
        count += 1
    return total / count if count else 0.0


def reset_loss_metric(loss_metric) -> None:
    """Reset a loss accumulator to zero."""
    try:
        import tensorflow as tf
        if isinstance(loss_metric, tf.keras.metrics.Metric):
            loss_metric.reset_states()
            return
    except (ImportError, AttributeError):
        pass
    if hasattr(loss_metric, "reset"):
        loss_metric.reset()


def get_loss_metric_result(loss_metric) -> float:
    """Return current accumulated mean from a loss metric/accumulator."""
    try:
        import tensorflow as tf
        if isinstance(loss_metric, tf.keras.metrics.Metric):
            return float(loss_metric.result().numpy())
    except (ImportError, AttributeError):
        pass
    if hasattr(loss_metric, "result"):
        return float(loss_metric.result())
    if hasattr(loss_metric, "avg"):
        return float(loss_metric.avg)
    raise AttributeError(f"Cannot get result from {type(loss_metric)}")


# ---------------------------------------------------------------------------
# DataLoader / tf.data.Dataset adapter
# ---------------------------------------------------------------------------

def count_batches(dataset_or_loader) -> int:
    """Count the number of batches in a dataset or DataLoader."""
    return sum(1 for _ in dataset_or_loader)


def first_batch_element_shapes(dataset_or_loader) -> List[tuple]:
    """Return the shape of each tensor in the first batch."""
    for batch in dataset_or_loader:
        if isinstance(batch, (list, tuple)):
            return [tuple(to_numpy(b).shape) for b in batch]
        return [tuple(to_numpy(batch).shape)]
    return []

