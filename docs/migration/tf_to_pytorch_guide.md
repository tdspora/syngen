# syngen: TensorFlow → PyTorch migration guide (for data scientists)

This is the **plain-language, comprehensive guide** to what changed when we moved
syngen's generative backend from **TensorFlow/Keras** to **PyTorch**. It explains
*what* changed, *why*, and *how the result was proven correct*.

> **One-line summary:** the synthetic-data model (a Conditional VAE) was rebuilt in
> PyTorch behind the **exact same public interface**. Training, generation, the CLI,
> the metadata format, and the MLflow logging all behave the same. The new model is
> proven to reproduce the old TensorFlow model's output distribution — *without the
> "distribution collapse" failure* — by an objective parity test suite.

**Companion documents in this folder**

| Document | What it is |
| --- | --- |
| `tf_to_pytorch_migration_plan.md` | The governing acceptance checklist (phases, invariants, the 6 collapse hypotheses). |
| `pytorch_backend_design.md` | The design decisions, grounded in `file:line` references to the old code. |
| `sign_off_records.md` | The dated phase-by-phase sign-off log with evidence. |
| **`tf_to_pytorch_guide.md`** (this file) | The friendly, all-in-one explainer. |
| `src/tests/integration/parity/README.md` | How to run the parity safety net. |

---

## Table of contents

1. [Why we did this](#1-why-we-did-this)
2. [What did NOT change (the contract)](#2-what-did-not-change-the-contract)
3. [What changed, at a glance](#3-what-changed-at-a-glance)
4. [The model: TF CVAE → PyTorch CVAE](#4-the-model-tf-cvae--pytorch-cvae)
5. [File-by-file changes](#5-file-by-file-changes)
6. [The "distribution collapse" problem and how we beat it](#6-the-distribution-collapse-problem-and-how-we-beat-it)
7. [Two real bugs the safety net caught](#7-two-real-bugs-the-safety-net-caught)
8. [Key fidelity decisions (and why)](#8-key-fidelity-decisions-and-why)
9. [The text / email path](#9-the-text--email-path)
10. [Artifacts (saved model files)](#10-artifacts-saved-model-files)
11. [Dependencies](#11-dependencies)
12. [The parity safety net (how we proved it works)](#12-the-parity-safety-net-how-we-proved-it-works)
13. [How to run things](#13-how-to-run-things)
14. [Results](#14-results)
15. [Post-migration verification & pre-existing fixes](#15-post-migration-verification--pre-existing-fixes)
16. [Known limitations & future ideas](#16-known-limitations--future-ideas)
17. [FAQ](#17-faq)

---

## 1. Why we did this

The model is a **Conditional Variational Autoencoder (CVAE)** that learns a table's
structure and then generates synthetic rows. It was written in TensorFlow/Keras.
We moved it to **PyTorch**.

The **non-negotiable success criterion** was *not* "the code runs" or "the loss goes
down." It was:

> **The generated data's per-column distribution must match the original** — an
> `age` column spanning 18–90 must not come out 18–40.

A previous PyTorch attempt had failed in exactly that dangerous way: it trained,
produced output, and **silently narrowed the distribution**. So the whole migration
was built around *proving* we didn't repeat that.

---

## 2. What did NOT change (the contract)

These were treated as **invariants**. If you used syngen before, none of this moves:

| Area | Guarantee |
| --- | --- |
| **CLI** | `launch_train` / `launch_infer` (and the `train` / `infer` / `syngen` console scripts) — identical signatures and behavior. |
| **Metadata** | The YAML structure and semantics are unchanged. |
| **Orchestration** | `worker.py`, `strategies.py`, `handlers.py`, `Dataset` semantics, reporters — unchanged. |
| **Generated-data semantics** | Row count = requested size; column order; dtype restoration; null/zero/datetime restoration; UUID regeneration; PK/UQ uniqueness; FK linkage — all preserved. |
| **MLflow logging** | `total_loss`, `kl_loss`, grouped `numeric/categorical/text` losses, and per-feature losses are still logged with the same names; `losses.csv` keeps the same columns. |

**The backend swap is invisible to the layers above it.** The strategy still asks for
a wrapper *by class name* (`VanillaVAEWrapper`) and the handler resolves it via
`globals()[name]` — so swapping the implementation behind that name required **zero**
changes to the worker/strategy/handler logic.

---

## 3. What changed, at a glance

Only the **backend** changed — five files — plus the dependency list and one test.

| File | Before (TF/Keras) | After (PyTorch) |
| --- | --- | --- |
| `ml/vae/models/features.py` | Keras `Input`/`Dense`/`LSTM`, Keras losses, Keras `Tokenizer` | Framework-neutral preprocessing **kept**; NN parts became a small PyTorch "spec"; **Keras-free** char tokenizer |
| `ml/vae/models/model.py` | Keras functional `Model` graph | `CVAEModule(nn.Module)` + a `CVAE` wrapper class |
| `ml/vae/models/custom_layers.py` | Keras `Layer` subclasses (mostly unused) | PyTorch `TextEncoder` / `TextDecoder` + `reparameterize` |
| `ml/vae/wrappers/wrappers.py` | `tf.data` + `GradientTape` loop | `DataLoader` + standard PyTorch training loop |
| `ml/handlers/handlers.py` | Keras `Tokenizer` in `LongTextsHandler` | The new Keras-free `CharTokenizer` |
| `setup.cfg` / `requirements.txt` | `tensorflow==2.15.*`, `keras==2.15.*` | `torch>=2.2` |
| `tests/unit/features/test_features.py` | `import tensorflow` for one test input | numpy softmax instead |

Everything else in `dataset.py` (detection, the pipeline, scalers wiring, key
handling) stayed put — it was already framework-neutral.

---

## 4. The model: TF CVAE → PyTorch CVAE

### The shape of the model (unchanged conceptually)

```text
inputs (one tensor per feature)
   │   each feature has its own small "encoder piece"
   ▼
[ concat ] → shared encoder: 3 × (Linear → BatchNorm → LeakyReLU → Dropout)
   ▼
mu , log_sigma                       ← the latent distribution
   ▼  reparameterize:  z = mu + exp(log_sigma/2)·eps
[ shared decoder ]: 3 × (Linear → LeakyReLU → Dropout)
   ▼
per-feature decoder "heads" → one reconstruction per feature
```

- **Numeric / date** features → a small dense head, **MSE** loss.
- **Categorical / binary** features → a softmax/sigmoid head, **cross-entropy** loss.
- **Text** features → an LSTM head, **sequence cross-entropy** loss.

### Generation (inference) path

Generation does **not** sample plain Gaussian noise. It reuses the original
mechanism:

```text
1. encode all training rows → collect their `mu` vectors
2. fit a BayesianGaussianMixture (BGM) on those `mu` vectors      ← "fit_sampler"
3. to generate N rows:  BGM.sample(N) → shuffle → shared decoder → inverse_transform
4. restore PK/UQ uniqueness, UUIDs, nulls, zeros, dates, FK links
```

> **Why a Gaussian mixture and not N(0,1)?** Because the latent space is **not**
> regularized toward N(0,1) — see [§6](#6-the-distribution-collapse-problem-and-how-we-beat-it).
> The BGM learns the actual shape of the encoder's output and samples from *that*.

### The single most important structural detail: **one shared decoder**

In the TF code the training model and the "generator" reused the **same** decoder
layers. In PyTorch we made this true *by construction*: there is **one** decoder
module, used both for reconstruction (fed `z`) and for generation (fed a BGM sample).

```python
class CVAEModule(nn.Module):
    def decode(self, latent):                 # ONE decoder, used by both paths
        d = self.dec_block2(self.dec_block1(self.dec_block0(latent)))
        return [head(d) for head in self.feature_heads]

    def forward(self, inputs):                 # reconstruction (training)
        mu, log_sigma = self.encode(inputs)
        z = reparameterize(mu, log_sigma)
        return self.decode(z), mu, log_sigma
```

This removes a whole class of bug ("the generator runs different/under-trained
weights than the model you trained").

---

## 5. File-by-file changes

### `features.py` — split "preprocessing" from "neural network"

Each feature class does two jobs. We **kept the first untouched** and **replaced the
second**:

| Job | Examples | Status |
| --- | --- | --- |
| **Preprocessing** (framework-neutral) | `fit` / `transform` / `inverse_transform`, scaler choice (`StandardScaler` / `MinMaxScaler` / `QuantileTransformer`), one-hot encoding, `is_positive`→`abs`, integer rounding, date↔timestamp | **Kept byte-for-byte** |
| **Neural network** (was Keras) | input shape, encoder piece, decoder head, loss | **Replaced** with a small PyTorch spec |

The new per-feature "spec" is just four things:

```python
class ContinuousFeature(BaseFeature):
    @property
    def encoded_dim(self): ...               # width it contributes to the encoder
    def build_encoder(self): ...             # nn.Module (identity for tabular features)
    def build_decoder_head(self, in_features): ...   # nn.Module producing the output
    def compute_loss(self, target, output): ...      # scalar loss
```

The per-feature recipes:

| Feature | Encoder piece | Decoder head | Loss |
| --- | --- | --- | --- |
| `ContinuousFeature` (numeric) | identity | `Linear(D,60)→ReLU→Linear(60,1)` (linear) | MSE |
| `DateFeature` | identity | same as numeric | MSE |
| `BinaryFeature` | identity | `Linear(D,1)→Sigmoid` | binary cross-entropy |
| `CategoricalFeature` | identity | `Linear(D,60)→ReLU→Linear(60,n_cat)→Softmax` | categorical cross-entropy |
| `CharBasedTextFeature` | `BiLSTM` | `Repeat→LSTM→Linear(vocab)` | softmax cross-entropy |
| `EmailFeature` | (extends text) | (extends text) | softmax cross-entropy |

> **Important:** features store **only plain data** (scalers, mappings, dimensions).
> All `nn.Module`s live on the model. That keeps the pickled `Dataset` light and
> backend-agnostic.

### `model.py` — `CVAEModule(nn.Module)` + `CVAE` wrapper

- `CVAEModule` builds the encoder/decoder/heads from the feature specs.
- `CVAE` keeps the friendly API the wrapper expects: `build_model`, `fit_sampler`,
  `sample`, `predict`, `save_state`, `load_state`, plus the unchanged
  `__make_pk_uq_unique` key-restoration logic.

We matched Keras' defaults exactly (these are subtle but they matter at low epoch
counts):

| Detail | Keras (old) | PyTorch (new) |
| --- | --- | --- |
| Dense weight init | `glorot_uniform`, bias `zeros` | `xavier_uniform_`, bias `zeros` |
| Encoder activation | `tf.nn.leaky_relu` (α=0.2) | `nn.LeakyReLU(0.2)` |
| Decoder activation | Keras `LeakyReLU()` (α=0.3) | `nn.LeakyReLU(0.3)` |
| BatchNorm momentum / eps | 0.99 / 1e-3 | **0.01** / 1e-3 (PyTorch momentum is the complement of Keras') |

### `custom_layers.py` — small PyTorch building blocks

The old Keras `FeatureLossLayer`/`SampleLayer` were **vestigial** (instantiated but
never wired into the active graph). They were replaced with what's actually used:

- `reparameterize(mu, log_sigma)` — the VAE sampling trick.
- `TextEncoder` — a `BiLSTM` over a one-hot character sequence.
- `TextDecoder` — `RepeatVector → LSTM → Linear`, producing per-position vocab logits.

### `wrappers.py` — the training loop

This is where `tf.data` + `GradientTape` became a standard PyTorch loop. Most of the
file (loss bookkeeping, MLflow logging, the `_restore_*` post-processing) was already
framework-neutral and was **reused unchanged**.

**Before (TF, simplified):**

```python
with tf.GradientTape() as tape:
    self.model(batch)                 # builds per-feature losses via add_loss
    loss = sum(self.model.losses)
optimizer.minimize(loss, self.model.trainable_weights, tape=tape)
```

**After (PyTorch, simplified):**

```python
self.optimizer.zero_grad()
recons, mu, log_sigma = self.model(batch)
loss = sum(feat.compute_loss(target, recon)
           for feat, target, recon in zip(features, batch, recons))
loss = loss + 0.0 * kl_divergence(mu, log_sigma)   # KL weight 0 — see §6
loss.backward()
self.optimizer.step()
```

Preserved exactly: **`Adam`** with `lr = 1e-4 · √(batch_size / 32)`, **early stopping**
(`min_delta=0.005`, `patience=10`, restore best weights), and the full MLflow/`losses.csv`
reporting.

### `handlers.py` — Keras-free long-text KDE

`LongTextsHandler` used a Keras `Tokenizer` only to count characters for a KDE. It now
uses our `CharTokenizer` — that was the **last** Keras import in the runtime.

---

## 6. The "distribution collapse" problem and how we beat it

**Collapse** = the model trains fine but the generated data is *narrower* than the
real data (age 18–90 → 18–40; a categorical column drops most of its categories).
The migration plan listed **six code-grounded ways** a PyTorch port could cause it.
Here's how each was handled:

| # | Risk | How we handled it |
| --- | --- | --- |
| 1 | **Re-enabling KL** squeezes the latent toward N(0,1) and narrows output | KL is computed but added with **weight 0**, exactly like TF (`add_loss(kl*0)`). Generation samples the fitted **BGM**, not N(0,1). |
| 2 | **Separate decoder** for generation runs different weights | **One** shared decoder module (see §4). Impossible to diverge. |
| 3 | **BatchNorm/Dropout mode mismatch** between train and generate | Train, encode, and generate all run in `eval()` (see §7). |
| 4 | **Feature order** scrambled between transform / decode / inverse | The `DataLoader` keeps the per-feature tuple in `Dataset.transform` order; the model's `ModuleList`s are in the same order. |
| 5 | **Scaler dropped/misapplied** squashes the range | Scalers and `inverse_transform` are the original neutral code, untouched. |
| 6 | **Wrong sampling** (wrong eps, or N(0,1) instead of the BGM) | `reparameterize` matches TF exactly (see §7); generation path unchanged. |

> **The guiding principle:** *the loss going down is not evidence of correctness.* The
> only acceptance evidence is **per-column statistical parity of generated data**
> against a frozen reference — with explicit checks that fail loudly on narrowing.

---

## 7. Two real bugs the safety net caught

These are the two issues that would have shipped a collapsed model, caught by the
parity tests.

### Bug A — training ran in "eval" mode in TF (and we had to match it)

This one is counter-intuitive, so read carefully.

- **"training mode" vs "eval mode"** is *not* about whether the model learns. It is a
  switch that only changes how **two** layer types behave in a forward pass:
  - **Dropout**: on in training mode, **off** in eval mode.
  - **BatchNorm**: uses the current batch's stats in training mode; uses **fixed
    stored stats** in eval mode.
  - The weights still update either way (learning happens via `loss.backward()`).

- **The old TF training loop called the model without `training=True`.** In Keras,
  that defaults to *inference* behavior. So TF trained with **Dropout off and
  BatchNorm frozen** — verified empirically (BatchNorm's running mean never moved;
  two forward passes were identical → dropout off).

- **Our job was to reproduce that model, not "fix" it.** So the PyTorch loop trains in
  `model.eval()`. When we instead used "proper" training mode, the model learned a
  *different* function that **collapsed the distribution** at generation time.

> **The deeper reason it matters:** the danger is a *mismatch* between training and
> generation. TF was self-consistent (both in eval mode), so it was safe. A half-fix
> (BatchNorm active in training but frozen at generation) creates the mismatch that
> collapses the output. We kept it consistent. This is **not** a TensorFlow bug — it's
> a classic missing-`training=True` gotcha in syngen's custom loop.

### Bug B — the reparameterization noise was too "rich" (real over-dispersion)

The VAE sampling step is `z = mu + exp(log_sigma/2)·eps`, with `eps ~ N(0,1)`.

- **TF** drew **one** `eps` vector of shape `(latent_dim,)` and broadcast it across the
  whole batch.
- Our first port drew a **fresh `eps` per row** — the textbook way, but it injected
  *more* noise during training. The decoder learned to spread its outputs wider to
  cope, and at generation it **over-dispersed**:

| Column | Real data | TF baseline | Port (per-row eps) | Port (per-batch eps, fixed) |
| --- | --- | --- | --- | --- |
| `age` | 18–90 | ~[33, 79] | **[2, 117]** ❌ (beyond the real range!) | **[32, 81]** ✅ |

Matching TF exactly (one `eps` broadcast across the batch) fixed it. This is precisely
the kind of subtle, distribution-distorting bug the parity suite exists to catch.

```python
# custom_layers.reparameterize — matches TF sample_z exactly
def reparameterize(mu, log_sigma):
    eps = torch.randn(mu.shape[-1])          # ONE (latent_dim,) vector
    return mu + torch.exp(log_sigma / 2.0) * eps   # broadcast across the batch
```

---

## 8. Key fidelity decisions (and why)

| Decision | Why |
| --- | --- |
| **KL weight = 0** | Matches TF; the latent is unregularized and generation uses the BGM. Re-enabling KL is the #1 collapse cause. |
| **One shared decoder** | Generation can't diverge from the trained decoder. |
| **Train in `eval()` mode** | Reproduces TF's (accidental) inference-mode training; keeps train/generate consistent. |
| **One `eps` broadcast per batch** | Matches TF; per-row eps over-disperses (Bug B). |
| **Glorot init, matched LeakyReLU slopes & BatchNorm momentum** | At low epoch counts the result is init-sensitive; matching Keras keeps the spread aligned. |
| **`drop_last=True`, `shuffle=False`** | Matches TF's `tf.data.batch(drop_remainder=True)` with no shuffle. |
| **Fixed training seed (42)** | Reproducible CPU runs (the determinism gate). Applied on the **train** path only, so it never overrides the infer-time `random_seed`. |
| **Versioned `vae_state.pt`** | Old TF checkpoints fail loudly instead of loading garbage. |

---

## 9. The text / email path

Text is the trickiest feature family. It was rebuilt **without Keras**:

- **`CharTokenizer`** replaces `keras.preprocessing.text.Tokenizer(lower=False,
  char_level=True)`. At char level Keras applies **no** punctuation filter and orders
  the vocabulary by **descending frequency** (ties by first appearance) — we replicate
  that exactly.
- **`pad_sequences`** → a numpy post-pad/truncate to `text_max_len` with value 0.
- **One-hot** → a numpy function that reproduces `tf.one_hot`, *including its quirk*
  that an out-of-range index becomes an all-zero vector (so generated-text statistics
  match the baseline).
- **Encoder / decoder** → `BiLSTM` encoder, `RepeatVector → LSTM → Linear` decoder.
- **Sampling** → numpy **top-p (nucleus)** filtering + multinomial draw, then map
  indices back to characters.
- **`EmailFeature`** keeps its behavior: take the name part, regenerate `name@domain`.
- **`LongTextsHandler`** uses `CharTokenizer` + the same scipy `gaussian_kde`.

> **Fun fact discovered during validation:** the PyTorch text model is actually *more
> faithful to the real data* than TF was. Source email names are ~8 chars
> (`alex3256`), so a realistic email is ~19 chars — the port produces ~18.5 (e.g.
> `mma94sa4@tdspora.ai`), while TF *over-generated* to ~25. (This is why the parity
> test treats text length as "must not collapse" rather than "must exactly match TF.")

---

## 10. Artifacts (saved model files)

| Artifact | Before | After |
| --- | --- | --- |
| Model weights | `vae.ckpt` + `vae_generator.ckpt` (Keras) | **`vae_state.pt`** — a single file with `{"backend": "pytorch", "version": 1, "state_dict": …}` |
| Latent sampler | `latent_model.pkl` (BGM) | unchanged |
| Dataset | `model_dataset.pkl` | unchanged |
| Long-text KDE | `kde_params.pkl` | unchanged (now produced without Keras) |
| Loss log | `…losses.csv` | unchanged schema |

**Loading is strict:** `load_state` checks the `backend` marker. A TensorFlow-era
`vae.ckpt` raises a **clear error** ("Retrain with the PyTorch backend") instead of
silently loading the wrong thing.

---

## 11. Dependencies

| Removed | Added |
| --- | --- |
| `tensorflow==2.15.*` | `torch>=2.2` |
| `keras==2.15.*` | — |

Also removed: the now-pointless `os.environ["TF_CPP_MIN_LOG_LEVEL"]` lines, and the
`import tensorflow` in `test_features.py` (its one use of `tf.nn.softmax` became a
numpy softmax). The `Dockerfile` installs from `requirements.txt`, so it picks up
`torch` automatically.

> **Note:** `torch>=2.2` installs the default wheel (which on Linux includes CUDA). A
> CPU-only image can install the `+cpu` build via PyTorch's CPU index — that's an
> install-time optimization, not a code change.

---

## 12. The parity safety net (how we proved it works)

Living under `src/tests/integration/parity/`, this is the evidence that the PyTorch
model behaves like the TF model.

### How it works

```text
1. make_fixtures.py  → 7 diverse datasets (numeric, categorical, text, email,
                        dates, UUID, PK/UQ/FK, a wide mixed table, a 4-table chain)
2. capture_baseline.py (run on TF) → for each fixture, run train→infer N=5 times,
                        record per-column statistical "bands" → baselines/*.json
3. test_parity.py    → run the PyTorch train→infer, profile the output, and assert it
                        falls inside the TF band (and that keys/FKs/dtypes hold)
```

### Why "bands" instead of a single reference (ensemble tolerances)

A generative model is **stochastic** — it produces somewhat different data every run,
especially on small tables. We discovered that a single TF baseline with tight ±25%
tolerances was **unsatisfiable even by TensorFlow** (two TF runs differed by more than
the bar).

So the gate is **calibrated to TF's own run-to-run variability**:

- Capture **N=5** TF runs per fixture; for each statistic record mean/std/min/max.
- A PyTorch value passes if it lands inside a **prediction interval**
  `mean ± t·std·√(1 + 1/N)` of the TF runs (a proper "where would a new sample fall?"
  band, which auto-widens when N is small).
- Plus **hard catastrophic backstops** that always apply: range coverage must be
  > 5 % and category coverage > 10 % (this is the real 18-90 → 18-40 guard).
- Plus **strict** key checks: PK/UQ uniqueness ≥ 0.999, FK validity ≥ 0.999.

> In short: **"PyTorch must behave like TF behaves with itself"** — fair to a faithful
> port, but still failing a genuine collapse. The collapse self-tests (which clamp a
> column on purpose) confirm the gate still trips.

### Failure → cause cheat-sheet

| Failure message | Likely cause |
| --- | --- |
| `RANGE COLLAPSE` on a numeric column | KL re-enabled, eval-mode missing, scaler dropped, or wrong eps |
| `CATEGORY COLLAPSE` / alien categories | feature-order mismatch, softmax head wrong size |
| `… outside TF band` (std/quantile) | over/under-dispersion vs TF |
| `uuid uniqueness < …` | UUID regeneration regressed |
| `FK validity < …` | FK KDE / key restoration regressed |
| `non-deterministic under fixed seed` | unseeded RNG, or train-mode dropout/BN active |

---

## 13. How to run things

### Train / infer (unchanged — this is the whole point)

```bash
train  --metadata_path metadata.yaml
infer  --metadata_path metadata.yaml --size 1000 --random_seed 10
# or programmatically:
python -c "from syngen.train import launch_train; launch_train(metadata_path='metadata.yaml')"
```

### Run the parity safety net

```bash
# fast, no backend needed — proves the collapse guard itself works:
pytest src/tests/integration/parity -m parity -k collapse --noconftest

# full statistical parity (needs the backend installed):
pytest src/tests/integration/parity -m parity
```

### Re-capture baselines (only if you change the model on purpose)

```bash
cd src/tests/integration/parity
N_RUNS=5 python capture_baseline.py        # run on the code you want as the reference
git add baselines/*.json
```

> ⚠️ The committed baselines were captured **once** on the TensorFlow code (the golden
> reference). You only re-capture if you deliberately change what "correct" means.

---

## 14. Results

| Gate | Result |
| --- | --- |
| **Statistical parity** | ✅ **All 7 fixtures pass, 0 discrepancies** (numeric_wide, categorical, text_email, datetime, keys, mixed_complex, relations_chain) |
| **Determinism** | ✅ 3/3 — repeated seeded CPU runs are stable |
| **Collapse self-tests** | ✅ still trip on an induced 18-90 → 18-40 collapse |
| **Unit suite** | ✅ 1096 passed, 0 failed |
| **Runtime** | ✅ no `tensorflow`/`keras` imports remain in `src/syngen` |
| **Packaging** | ✅ `pip check` clean; `train`/`infer`/`syngen` entry points resolve |

**Bottom line:** the PyTorch model's deviation from the TF baseline equals
TensorFlow's *own* run-to-run deviation — i.e. it behaves like TF — and it does **not**
collapse the distribution.

---

## 15. Post-migration verification & pre-existing fixes

After the backend swap was green, the **CLI** and the **orchestration layer**
(worker → strategy → handler → dataset → reporters) were exercised end-to-end on
real data. The migration itself touched almost none of this layer — `worker.py`,
`dataset.py`, `configurations.py`, and `validation_schema.py` are **unchanged**;
only `strategies.py` (a removed dead env line) and `handlers.py` (the Keras-free
tokenizer, plus the two fixes below) changed. The verification confirmed the layer
works, and surfaced **three pre-existing bugs** that were *not* caused by the
migration (they exist on `main`) but were exposed by finally running these paths.

### 15.1 What was verified end-to-end

| Scenario | What it exercises | Result |
| --- | --- | --- |
| Single table (`housing.csv`, 20 640 rows) | numeric + categorical, scalers, null drop | ✅ distributions match (see §below) |
| 2-table FK (`housing_properties` → `housing_conditions`) | PK uniqueness, FK generation, multi-table order | ✅ PK 100% unique, FK 100% valid |
| 4-table chain (`relations_chain`) | a chain `regions ← stores ← sales` **and** `sales → products` (two FKs on one table) | ✅ all PKs unique; all 3 FKs **100%** valid |
| `reports: all` | `SampleAccuracyReporter` + `AccuracyReporter` + metrics | ✅ accuracy & sample HTML reports generated per table |
| CLI: `train`, `infer`, `syngen` | click arg parsing, both `--source/--table_name` and `--metadata_path` modes | ✅ (after fix A) |
| Data formats: CSV / **Avro** / **Excel** | `DataLoader` read (source) and write (destination) | ✅ all three, source and destination |
| `run_parallel: true` | `pathos` multiprocessing inference | ✅ (after fixes B & C) |

Example — the 4-table chain FK check on generated data:

```text
regions:   8 rows | PK unique: True
products: 30 rows | PK unique: True
stores:   40 rows | PK unique: True
sales:   600 rows | PK unique: True
FK stores.region_id -> regions.region_id  : 100.0%
FK sales.store_id   -> stores.store_id     : 100.0%
FK sales.product_id -> products.product_id : 100.0%   (2nd FK on sales)
```

Example — Avro/Excel as both source and destination:

```yaml
housing_avro:
  train_settings: {source: /path/housing.avro}      # read Avro
  infer_settings: {destination: /path/out.avro}      # write Avro
housing_xlsx:
  train_settings: {source: /path/housing.xlsx}      # read Excel
  infer_settings: {destination: /path/out.xlsx}      # write Excel
```
→ both train and generate correctly (verified output shape 100 × 10).

### 15.2 Three pre-existing bugs found and fixed

> ⚠️ All three are **pre-existing on `main`** and independent of the TF→PyTorch
> swap. They surfaced only because the CLI/`run_parallel` paths were finally run
> end-to-end. Good candidates to cherry-pick to `main` on their own.

#### Fix A — CLI `train`/`infer` ignored all arguments

**Symptom**
```text
$ train --source examples/example-data/housing.csv --table_name housing_test
AttributeError: 'NoneType' object has no attribute 'keys'
```

**Root cause.** `train.py` has two functions: `launch_train(...)` (the
**programmatic API** — plain Python kwargs, used by tests/SDK) and
`cli_launch_train(...)` (the **click command** that parses `--source` etc.). The
`setup.cfg` console script pointed at the *plain* one:

```ini
train = syngen.train:launch_train      # ← parses nothing; argv ignored
```

So setuptools' wrapper called `launch_train()` with **no arguments**; everything
defaulted to `None` and it crashed in schema validation.

**Fix.** Point the entry points at the click commands:

```ini
train = syngen.train:cli_launch_train
infer = syngen.infer:cli_launch_infer
```

> 🔁 Requires a reinstall (`pip install -e .` / `pip install .`) to regenerate the
> `bin/` console scripts.

#### Fix B — `run_parallel=True` crashed

**Symptom**
```text
AttributeError: 'VaeInferHandler' object has no attribute 'random_seed_list'.
Did you mean: 'random_seeds_list'?
```

**Root cause.** A typo: the attrs field was declared `random_seed_list` (singular)
but every use is `random_seeds_list` (plural), so the declared field was never
assigned. With `run_parallel=True`, `pool.map` pickles the handler (via `dill`) to
send it to worker processes, which touches the unset field → `AttributeError`. The
sequential path never pickles the handler, so it never hit this.

**Fix.** Rename the declaration to `random_seeds_list` (matching all uses).

#### Fix C — `run_parallel=True` generated too many rows

**Symptom.** With an explicit `batch_size < size`, `run_parallel` produced
`size × batch_num` rows:

```text
size=400, batch_size=100  →  1600 rows   (expected 400)
```

**Root cause — double batching.** `handle()` loops over `batch_num` batches and
calls `run(batch, run_parallel)` for each. But `run()`'s parallel branch
**re-splits the full `self.size`** itself (`pool.map` over `split_by_batches()`),
so each of the `batch_num` outer iterations regenerated *all* batches. (With the
default `batch_size`, `batch_num == 1`, so this stayed hidden.)

**Fix.** On the parallel path, let `run()` do the batching **once**; the sequential
path is unchanged:

```python
if self.run_parallel:
    prepared_batches.append(self.run(self.size, run_parallel=True))   # parallel: one call
else:
    for i, batch in enumerate(self.split_by_batches()):               # sequential: loop (unchanged)
        prepared_batches.append(self.run(batch, run_parallel=False))
```

**After fixes B & C:**

| Scenario | Rows |
| --- | --- |
| `run_parallel` + default `batch_size` | 400 ✅ |
| `run_parallel` + `batch_size=100, size=400` | 400 ✅ (was 1600) |
| sequential + `batch_size=100, size=400` | 400 ✅ (unchanged) |

All handler/worker/launcher unit tests still pass (252), and the full unit suite
stays at 1096 passed.

---

## 16. Known limitations & future ideas

- **Ensemble-gate tolerances are calibration choices.** Values like the prediction
  interval level (`pi_alpha=0.002`) and the text "factor of 2" rule are recorded in
  every baseline JSON and are open to review/adjustment.
- **Baselines are a one-time TF snapshot.** TF training was unseeded, so the reference
  reflects one particular TF run's behavior (within the ensemble band).
- **Conditioning (`is_cond`) was not ported** — it was never exercised in the codebase.
  There's a clear extension point if it's ever needed.
- **Optional CPU-only torch wheel** for slimmer Docker images.
- 💡 **Possible model *improvement* (separate from this migration):** the encoder's
  `BatchNorm` and `Dropout` layers are currently inert (a TF leftover — see §7). Making
  them actually active during training is the textbook design and *might* improve the
  generator — but it's a **model change, not a migration**: it would no longer match the
  TF baseline, so it needs its own evaluation and a fresh baseline. Good candidate for a
  follow-up ticket.

---

## 17. FAQ

**Q: Did the old TensorFlow implementation "work," or was it set up wrong?**
Both, depending on the lens. It *worked* — it shipped and generated usable data — and
it was internally consistent. But it had a *latent bug*: the encoder's Dropout and
BatchNorm layers weren't doing their job (a missing `training=True` in syngen's custom
loop). Harmless in practice, because training and generation were consistent. See §7.

**Q: Is the eval-mode-training thing a TensorFlow bug?**
No. TensorFlow behaved as documented. The missing `training=True` flag was in
**syngen's** wrapper code — a very common TF2 custom-loop gotcha.

**Q: Will my existing TensorFlow model checkpoints still load?**
No — and that's intentional. They fail with a clear error. Retrain with the PyTorch
backend (retraining is cheap and expected).

**Q: Is the generated data identical to before?**
Not bit-for-bit (different framework, different RNG) — and it shouldn't be. It's
*statistically equivalent*: within TensorFlow's own run-to-run variability, with no
distribution collapse.

**Q: Why does generation use a Gaussian mixture instead of plain noise?**
Because KL is disabled, the latent space isn't shaped like N(0,1). The
`BayesianGaussianMixture` learns the encoder's actual output shape and samples from
that — matching the original design.

**Q: Where's the authoritative phase-by-phase record?**
`docs/migration/sign_off_records.md` — dated Accepted records with evidence for every
phase (A through J).
