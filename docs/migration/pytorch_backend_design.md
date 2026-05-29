# PyTorch backend design (Phases C–F implementation note)

Companion to `tf_to_pytorch_migration_plan.md`. This records the concrete design
decisions for the PyTorch port, each grounded in the current TF code, and how
each maps onto the six collapse hypotheses. Citations are `file:line` against the
TF code being replaced.

## Boundary (Phase C)

Orchestration is untouched. `strategies.py:89,188` instantiate the wrapper by
class **name** (`VanillaVAEWrapper.__name__`); `handlers.py:68` resolves it via
`globals()[cls_name]`, populated by `from syngen.ml.vae import *`
(`handlers.py:21`). So the swap is transparent as long as we keep:

- `syngen.ml.vae` exporting `BaseWrapper`, `VAEWrapper`, `VanillaVAEWrapper`
  with identical constructor signatures (`wrappers.py:566`).
- The `BaseWrapper` ABC: `fit_on_df`, `predict_sampled_df`, `save_state`
  (`wrappers.py:42-52`), plus `load_state` used at `wrappers.py:602`.

Everything in `wrappers.py` that is pure-Python loss bookkeeping is **reused
unchanged**: `_calculate_loss_by_type`, `_get_grouped_losses`,
`_get_mean_feature_losses`, `_get_ending`, `_monitor_feature_losses`,
`_fetch_feature_losses_info`, `_accumulate_feature_losses`, `_update_losses_info`,
`__save_losses`, `_gather_losses_info`, `_log_losses_info_to_mlflow`
(`wrappers.py:197-358`), and the post-processing restorers + `predict_sampled_df`
(`wrappers.py:106-149,505-537`). This preserves the MLflow loss contract
(`total_loss`, `kl_loss`, grouped `numeric/categorical/text`, per-feature) and the
`losses.csv` schema by construction.

## Feature contract split (Phase D)

`features.py` already separates **neutral preprocessing** (`fit`, `transform`,
`inverse_transform`, `_select_scaler`, `_get_quantile_transformer_params`, the
tokenizer, top-p/top-k) from **framework NN** (`input`, `encoder`,
`__decoder_layer`, `create_decoder`, `loss`). The port keeps the neutral half
byte-for-byte and replaces the NN half with framework-neutral *specs* + small
builders the model assembles:

| Feature | `encoded_dim` | encoder piece | decoder head | loss |
| --- | --- | --- | --- | --- |
| `ContinuousFeature` | `input_dimension` (1) | identity | `Linear(D,60)+ReLU → Linear(60,input_dim)` linear (`features.py:290-305`) | MSE (`:329`) |
| `DateFeature` | `input_dimension` (1) | identity | same as Continuous (`:760-780`) | MSE (`:804`) |
| `BinaryFeature` | `input_dimension` | identity | `Linear(D,input_dim)` **sigmoid**, no hidden (`:142`) | BCE (`:154`) |
| `CategoricalFeature` | `input_dimension` (= n_cat) | identity | `Linear(D,60)+ReLU → Linear(60,n_cat)` **softmax** (`:411-428`) | categorical CE (`:454`) |
| `CharBasedTextFeature` | `2*rnn_units` (BiLSTM) | `BiLSTM(rnn_units)` (`:612`) | `Repeat(text_max_len) → LSTM(rnn_units) → Linear(rnn_units,vocab)` logits (`:618-623`) | softmax CE over vocab (`:645`) |

`weight_randomizer` defaults to `(1,1)` for every feature, so `random_weight` is a
fixed scalar `1.0` (`features.py:179,327`); sampled once at build to match. The
features store **only picklable data** (scalers, mappings, dims) — all `nn.Module`
instances live on the model, so the `Dataset` pickle (`wrappers.py:99-104`, taken
before the model is built) stays light and backend-agnostic, exactly as today
where the lazy Keras tensors are not yet materialized at pickle time.

## Data path (Phase E)

`Dataset.transform` returns a **list of per-feature arrays in
`self.features.items()` order** (`dataset.py:1118-1127`); `inverse_transform`
zips predictions back with `self.features.items()` in the *same* order
(`dataset.py:1138-1158`). That order is the load-bearing invariant (collapse
hypothesis #4). The torch data path preserves it:

- A `Dataset` wrapping the transform output as tensors, `__getitem__` returning a
  tuple in feature order; default collate stacks each position → a tuple of
  `(batch, …)` tensors still in feature order.
- `DataLoader(batch_size, shuffle=False, drop_last=True)`. **`drop_last=True`**
  matches `tf.data … .batch(batch_size, drop_remainder=True)`
  (`wrappers.py:469`). **`shuffle=False`** matches TF: the current pipeline does
  not call `.shuffle()` (`wrappers.py:455-469`). Both are contract items.
- Seeding: Python / `numpy` / `torch` seeded together; CPU-only for wave 1.

## Model & loop (Phase F)

`CVAEModule(nn.Module)` mirrors `model.py:66-176`:

- **Encoder**: concat per-feature encoder pieces → 3 × `[Linear → BatchNorm1d →
  LeakyReLU → Dropout(0.2)]` → `mu`, `log_sigma` heads (`model.py:138-156`).
- **Reparameterization**: `z = mu + exp(log_sigma/2)·eps`, `eps ~ N(0,1)`
  (`model.py:56-59`). We draw `eps` shaped `(batch, latent_dim)` (standard);
  TF draws `(latent_dim,)` and broadcasts one sample across the batch — an
  immaterial difference because KL weight is 0 (so `log_sigma → −∞`, `z ≈ mu`)
  **and** generation never uses `z` (it samples the fitted BGM on `mu`). Noted as
  an approved, documented deviation.
- **Shared decoder**: a single decoder (3 × `Linear+LeakyReLU+Dropout`,
  `model.py:158-172`) + the per-feature heads, used for **both** reconstruction
  (`z`) and generation (BGM latent). In TF the sharing is achieved by re-applying
  the same Keras layer instances to two inputs (`model.py:170-176`) and by the
  `@lazy` per-feature `__decoder_layer` cache so `create_decoder` reuses the same
  head on both calls (`features.py:289-318`). In torch this is automatic: there is
  **one** decoder module; `decode(z)` and `decode(latent)` call the same
  parameters. Directly defeats collapse hypothesis #2 — no separate
  `generator_model`, no chance of loading generator weights into the wrong module.
- **KL**: computed as `0.5·Σ(exp(log_sigma)+mu²−1−log_sigma)` (`model.py:103-107`)
  and **added with weight 0** (`model.py:129`). This is the prime collapse
  suspect (#1): re-enabling it squeezes the posterior toward N(0,1), narrowing the
  BGM manifold → range collapse. We log the true KL value under `kl_loss` (more
  informative than TF's logged `0`; the name/contract is preserved) but it
  contributes 0 to the optimized total.
- **eval() discipline** (#3): `fit_sampler` encoding (`model.py:182-196`) and
  `sample` decoding (`model.py:203-208`) run under `module.eval()` +
  `torch.no_grad()` so BatchNorm uses running stats and Dropout is off; the
  training loop is the only `module.train()` region.
- **Latent sampling** (#6): keep `BayesianGaussianMixture(n_components, n_init=10)`
  fit on encoder `mu` (`model.py:193`); `sample(n)` → `np.random.shuffle` →
  `decode` → `inverse_transform` → `__make_pk_uq_unique` (`model.py:203-212,257`).

**Training loop** (`wrappers.py:360-491`): standard PyTorch — `Adam(lr =
1e-4·√(batch_size/32))` (`wrappers.py:448`), `loss.backward()`,
`optimizer.step()`; per-feature loss = mean over batch, total = Σ per-feature
(+ 0·KL); early stopping `min_delta 0.005`, `patience 10`, best `state_dict`
checkpointed to a temp file and restored on stop (`wrappers.py:363-433`).

## Artifacts (Phase H)

New `vae_state.pt` = `{"backend": "pytorch", "version": <n>, "state_dict": …}` +
the existing `latent_model.pkl`. `load_state` asserts the backend marker and
raises a clear error for TF-era `vae.ckpt`/`vae_generator.ckpt` (no silent
fallback). `model_dataset.pkl` stays.
