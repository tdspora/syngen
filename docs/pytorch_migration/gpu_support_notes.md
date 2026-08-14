# GPU support for train/infer — findings, changes, and how it works

Status: implemented, committed, and validated on real GPU hardware (4x NVIDIA
L4). A subsequent merge from `main` silently reintroduced a regression in one
of the Phase 3 changes (see "Real bug #5" below) — that regression has since
been found and fixed, and this doc has been updated to reflect the current,
working state. Companion to `tensorflow_to_pytorch.md` section 8.8 in this
same directory, which carries the durable summary — this file is the full
narrative: what was actually built, what was discovered along the way that
the original design plan didn't anticipate, and how to use/understand it as
a developer or as a regular CLI/SDK user.

## TL;DR for users

**You don't need to do anything differently.** `train`/`infer` (CLI, SDK, or
metadata-driven multi-table runs) automatically use a GPU if one is visible,
and fall back to CPU if not. There is no new flag, no new SDK parameter, no
metadata field. The exact same command:

```bash
train --metadata_path metadata.yaml
infer --metadata_path metadata.yaml
```

now runs faster on a GPU box with zero changes to how you invoke it. If you
have multiple tables in one metadata file and multiple GPUs, training
distributes different tables across different GPUs automatically. If you're
generating a lot of rows for one table with `run_parallel: true`, inference
batches get distributed across GPUs the same way.

The only thing that changed under the hood: torch now actually moves the
model and data onto `cuda:0`, `cuda:1`, etc. when available.

## TL;DR for developers

Three independent, incrementally-shippable phases landed (see
`tensorflow_to_pytorch.md` section 8.8 for the durable summary and the
original design rationale):

1. **Phase 0** — one bugfix (device-unsafe loss accumulator).
2. **Phase 1** — device auto-detection + single-GPU path. Fully self-contained,
   zero multiprocessing.
3. **Phase 2** — multi-GPU **task-parallel** training: one whole table per GPU,
   not DistributedDataParallel.
4. **Phase 3** — multi-GPU task-parallel inference, reusing the existing
   CPU-multiprocessing pool machinery.

Multi-GPU here means **task parallelism across tables/batches**, never
splitting one model's batch across GPUs. A single large table's *training*
still uses exactly one GPU no matter how many are installed — see
"Why not DDP" below for the reasoning.

**The most valuable part of this work wasn't the design — it was actually
running it.** Four real bugs were found only by executing training/inference
on this sandbox's 4 GPUs; none were predicted by the pre-existing migration
docs or the initial design plan. A fifth real bug was found afterward, not on
GPU hardware but via a basic post-merge smoke test — it broke every
non-parallel `infer` call on *every* device, including CPU. Every one of
these five would have shipped broken if not caught. Details in "Real bugs
found" below.

---

## Why not DistributedDataParallel

Decided with the user before implementation, worth restating here because
it's the single biggest architectural choice in this work:

- The CVAE is architecturally small (a handful of Linear/BatchNorm blocks,
  default `batch_size=32`). DDP's per-step gradient-sync overhead can exceed
  the compute it saves for a model this size — the classic "DDP made it
  slower" failure mode.
- Most of syngen's end-to-end wall-clock isn't in the neural net at all: the
  `BayesianGaussianMixture` fit, preprocessing, and report generation are
  scikit-learn/pandas and stay on CPU regardless of what happens to the
  training loop. A perfect DDP speedup on the NN training loop still wouldn't
  translate to a proportional end-to-end speedup.
- Task-parallel-across-tables is simpler, matches the existing
  sequential-per-table architecture, and gives real speedup for the actual
  common case: multiple tables in a metadata-driven run.

**Consequence to understand:** if you have **one huge table** and **8 GPUs**,
training that one table still only uses **1 GPU**. If you have **8 tables**
and **8 GPUs**, training uses all 8, one table per GPU. Inference is less
constrained — a single table's generation batches can spread across all
visible GPUs regardless of table count, because inference already had a
batch-splitting mechanism (`run_parallel`) that this work extended.

---

## Phase 0 — Bugfix: device-unsafe loss accumulator

**File:** `src/syngen/ml/vae/wrappers/wrappers.py`

```python
# Before (always created on the CPU default device):
recon_total = torch.zeros((), dtype=recons[0].dtype)

# After:
recon_total = torch.zeros((), dtype=recons[0].dtype, device=recons[0].device)
```

This was documented as a known blocker in
`docs/pytorch_migration/tensorflow_to_pytorch.md` §8.8.2 — the migration
authors correctly predicted it would fail on the very first GPU run, but had
no way to verify it since they never had GPU hardware. It did fail on the
very first GPU run here, exactly as predicted.

**Regression test:** `test_train_step_accumulator_matches_recon_device`
(`src/tests/unit/wrappers/test_wrappers.py`) spies on `torch.zeros` and
asserts an explicit `device=` kwarg is always passed. This catches the bug
class (missing `device=`) without needing GPU hardware — on CPU-only CI the
accumulator's device happens to already match by coincidence (both are
`cpu`), so the *only* way to actually catch this bug is to check that the
kwarg is passed at all, not to compare devices after the fact.

---

## Phase 1 — Device auto-detection

### New module: `src/syngen/ml/utils/device.py`

The entire public surface added by this work. Four small functions:

```python
from syngen.ml.utils.device import cuda_device_count, gpu_available, assign_gpu_index, resolve_device

cuda_device_count()      # -> 0 if no CUDA, else torch.cuda.device_count()
gpu_available()          # -> cuda_device_count() > 0
assign_gpu_index(2)      # round-robin: with 3 GPUs, index 2 -> 2; index 5 -> 2 again
resolve_device(None)     # -> torch.device("cpu")
resolve_device(1)        # -> torch.device("cuda:1") if CUDA available, else cpu (safe fallback)
```

**Important implementation note for developers:** this module is imported
*directly* (`from syngen.ml.utils.device import ...`), never re-exported
through `syngen/ml/utils/__init__.py`. Reason: `train.py`/`infer.py` call
`limit_thread_parallelism()` (sets `OMP_NUM_THREADS`/`MKL_NUM_THREADS`)
*before* importing anything that pulls in `torch`, specifically so those env
vars are set before torch/MKL initializes. Re-exporting `device.py` from
`utils/__init__.py` would transitively import torch too early and silently
break that ordering. If you add a new function to `device.py`, keep it out of
`utils/__init__.py`.

### Where device flows through

```
CVAE.__init__(device=None)              # None -> auto: resolve_device(assign_gpu_index(0))
  -> self.device                        # e.g. cuda:0, or cpu if no GPU
  -> build_model(): CVAEModule(...).to(self.device)
  -> fit_sampler / predict / sample:    tensors created with device=self.device

VAEWrapper.device (new dataclass field, default None)
  -> VanillaVAEWrapper.__init__ passes device=self.device into CVAE(...)
  -> then does self.device = self.vae.device (adopts the resolved device)
  -> _train_step moves each batch to self.device before the forward pass
```

**Example — what actually changed for a developer reading `model.py`:**

```python
# CVAE is constructed exactly like before - no caller needs to change:
vae = CVAE(dataset, batch_size=32, latent_dim=10, intermediate_dim=128, latent_components=30)
print(vae.device)  # cuda:0 on a GPU box, cpu otherwise - decided automatically

# To force CPU even on a GPU box (useful for debugging/reproducing a CPU-only
# result), pass device explicitly - this parameter exists but nothing in the
# CLI/SDK exposes it, by design (no new public API surface):
vae = CVAE(dataset, batch_size=32, latent_dim=10, intermediate_dim=128,
           latent_components=30, device=torch.device("cpu"))
```

### Real bug #1 found on GPU hardware: cuDNN LSTM backward restriction

Training deliberately runs the *whole model* in `.eval()` mode (a documented,
intentional choice — it freezes BatchNorm running stats and turns off
Dropout, matching the original TensorFlow behavior exactly). This works fine
on CPU. On a real GPU, the very first table with a text/long-text column
(anything using the BiLSTM `TextEncoder`/`TextDecoder` in
`custom_layers.py`) crashed:

```
RuntimeError: cudnn RNN backward can only be called in training mode
```

cuDNN's fused LSTM kernel refuses `.backward()` on a module left in `.eval()`
mode — a restriction that simply doesn't exist in PyTorch's CPU LSTM
implementation, so nothing in months of CPU testing could have surfaced it.

**Fix** (`wrappers.py`, new helper `_unfreeze_lstm_submodules`):

```python
self.model.eval()                    # freezes BatchNorm/Dropout, as intended
_unfreeze_lstm_submodules(self.model)  # but LSTM submodules go back to train()
```

Why this is safe: the LSTMs in `custom_layers.py` are single-layer with no
inter-layer dropout, so `train()`/`eval()` has **zero effect on their actual
math** — only on cuDNN's internal bookkeeping needed for backward. Everything
else (BatchNorm, standalone `nn.Dropout` blocks) stays in `eval()` exactly as
before.

**Regression test:** `test_unfreeze_lstm_submodules_restores_train_mode_on_lstm_only`
builds a tiny `Sequential(LSTM, Dropout, BatchNorm1d)`, calls `.eval()`, runs
the helper, and asserts only the LSTM flipped back to `train()`.

### Real bug #2 found on GPU hardware: OOM from unbatched full-table forward pass

`CVAE.fit_sampler` (used after training, to fit the latent-space Gaussian
mixture) and `CVAE.predict` both ran the **entire table** through the model
in a single forward pass — fine when everything lives in system RAM, but on
the `long_text_table` benchmark dataset this tried to allocate:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 26.67 GiB.
```

A table with a free-text column one-hot-encodes/tokenizes into a large
per-row tensor; multiply by 20,000 rows and it doesn't fit on a 22GB L4 GPU
in one shot.

**Fix** (`model.py`, new helpers `_batched_encode`/`_batched_forward`):

```python
# Before:
mu, _ = self.model.encode(_to_tensors(transformed_data, self.device))

# After:
mu, _ = _batched_encode(self.model, _to_tensors(transformed_data, self.device), self.batch_size)
```

Chunks the input by `self.batch_size` and concatenates results. Why this is
safe: eval-mode `BatchNorm` uses running statistics (a fixed affine transform
independent of what's in the current batch), so chunking never changes the
*output* — it only bounds *peak memory* to one batch's worth of activations
instead of the whole table's.

**Regression tests:** `TestBatchedEncodeForward` in `test_model.py` uses tiny
deterministic fake models (`mu = 2*x`, `recon = 3*x`) and asserts the batched
and unbatched results are bit-identical.

### Determinism note

`_seed_everything` now also seeds CUDA when available:

```python
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Gated behind `torch.cuda.is_available()`, so the CPU path is untouched.
**Caveat worth knowing:** GPU floating-point reduction order differs from
CPU, so seeding does *not* make GPU and CPU runs bit-identical — only
same-device runs are reproducible. If you train the same table twice on the
same GPU with the same seed, you get the same result; if you train once on
CPU and once on GPU, don't expect byte-for-byte identical output (distributions
should still match — see the accuracy numbers in "Real validation" below).

---

## Phase 2 — Multi-GPU task-parallel training

### How it decides to go parallel

`Worker._should_train_in_parallel`:

```python
def _should_train_in_parallel(self, tables_for_training) -> bool:
    return (
        self.loader is None            # no custom SDK loader (picklability)
        and len(tables_for_training) > 1
        and cuda_device_count() > 1
    )
```

**Example:** a metadata file with 4 tables, run on a box with 4 GPUs:

```yaml
# metadata.yaml
customers:
  train_settings: { source: "customers.csv" }
  keys: { customer_id_pk: { type: "PK", columns: [customer_id] } }
products:
  train_settings: { source: "products.csv" }
orders:
  train_settings: { source: "orders.csv" }
  keys:
    order_id_pk: { type: "PK", columns: [order_id] }
    customer_id_fk: { type: "FK", columns: [customer_id], references: { table: customers, columns: [customer_id] } }
payments:
  train_settings: { source: "payments.csv" }
```

```bash
train --metadata_path metadata.yaml
```

With 4 tables and 4 GPUs, this trains all 4 concurrently, one table per GPU
(customers→cuda:0, products→cuda:1, orders→cuda:2, payments→cuda:3 — actual
assignment is round-robin by dispatch order, not by name). With 1 table or 1
GPU, it silently falls back to the exact same sequential loop as before this
change — **zero behavioral risk for the overwhelming majority of current
users and all current CI**, since that path is unreachable without >1 GPU.

Note `orders` has both a PK and an FK — this internally gets split into
`orders_pk`/`orders_fk` surrogate tables by pre-existing logic
(`Worker._split_pk_fk_metadata`), so the actual parallel dispatch in this
example handles 5 jobs, not 4. This was deliberately exercised in the
relational benchmark dataset used for the initial validation pass,
specifically to catch bugs in that interaction — and it did (see below).

### What actually happens per table (`_run_train_table_job`)

Each table becomes a `_TrainTableJob` (a picklable `NamedTuple`) dispatched to
a `multiprocessing.get_context("spawn").Pool`. **`spawn`, not `fork`** — CUDA
contexts can't be safely forked, and `spawn` gives every worker a completely
fresh Python interpreter.

That "fresh interpreter" property is exactly what caused the next two real
bugs.

### Real bug #3 found on GPU hardware: ProgressBarHandler crash

```
TypeError: unsupported operand type(s) for /: 'NoneType' and 'int'
```

`ProgressBarHandler` is a singleton, but singletons don't survive `spawn` —
each child process gets its own, uninitialized one. The training loop
(`VAEWrapper._train`) does `ProgressBarHandler().delta / (epochs * 2)`
directly — in the sequential path, the parent always calls
`ProgressBarHandler().set_progress(delta=...)` before training starts, so
`.delta` is never `None`. In the parallel path, nothing had done that inside
the fresh child.

**Fix:** `_TrainTableJob` now carries a `delta` field, and
`_run_train_table_job` calls `ProgressBarHandler().set_progress(delta=job.delta, ...)`
as its very first line, before touching `TrainStrategy`.

**Accepted side effect, not a bug:** per-epoch progress bar granularity is
local to each child process and invisible to the parent. If you're polling
`ProgressBarHandler().info` for a live progress UI, you'll see the coarse
per-table start/complete updates but not the fine per-epoch ones during the
parallel phase. This mirrors an identical, already-accepted trade-off in the
pre-existing CPU inference pool (its own comment literally warns about
duplicate log lines for the same reason).

### Real bug #4 found on GPU hardware: report generation reads a file that isn't there yet

```
FileNotFoundError: [Errno 2] No such file or directory:
'model_artifacts/tmp_store/products/input_data_products.pkl'
```

Same root cause category (singletons don't survive `spawn`), different
symptom. `Report()` is also a singleton — reporters get registered against
whichever process's `Report()` instance is running
(`TrainStrategy.add_reporters()`, called deep inside `TrainStrategy().run()`).
So `Report().generate_report()` has to run **inside the same child process**
that did the training, not in the parent afterward (the parent's `Report()`
instance never saw those registrations).

The bug: the sample report's `_extract_report_data()` reads back a file
(`input_data_{table}.pkl`) that a *different* method,
`Worker._save_input_data`, writes. In the sequential path, that write happens
right after training, well before any report generation. When report
generation moved into the child, it initially ran *before* that write (which
was still happening in the parent, one full `imap_unordered` iteration
later) — so the report tried to read a file that didn't exist yet.

**Fix:** a new module-level `_save_input_data_for_job` (a pure-function
mirror of `Worker._save_input_data`, callable without a `Worker` instance)
runs inside `_run_train_table_job`, in the correct order:

```python
TrainStrategy().run(...)                              # train
_save_input_data_for_job(job.metadata, job.data, job.table)  # write input snapshot
Report().generate_report()                             # THEN generate report (reads it back)
Report().clear_report()
```

**Regression test:** `test_run_train_table_job_saves_input_data_before_generating_report`
mocks both calls and asserts the order with a shared `call_order` list —
this is the kind of ordering bug that's easy to reintroduce silently if
someone refactors this function later, so the test pins the order explicitly
rather than just checking both calls happened.

### Files touched (Phase 2)

- `src/syngen/ml/config/configurations.py` — `TrainConfig.device` field.
- `src/syngen/ml/strategies/strategies.py` — forwards `device` into `VaeTrainHandler`.
- `src/syngen/ml/handlers/handlers.py` — `VaeTrainHandler.device` field, `create_wrapper` forwards it.
- `src/syngen/ml/worker/worker.py` — `_TrainTableJob`, `_run_train_table_job`, `_save_input_data_for_job`, `_should_train_in_parallel`, `__train_tables_parallel`.

---

## Phase 3 — Multi-GPU task-parallel inference

Smaller change — it extends the *existing* CPU multiprocessing pool in
`VaeInferHandler` rather than building a new one.

### `fork` → `spawn`, but only when it matters

```python
def _select_mp_start_method() -> str:
    if sys.platform == "win32" or cuda_device_count() > 0:
        return "spawn"
    return "fork"
```

Why "any GPU visible" and not "will this pool use >1 GPU": the *parent*
process might have already touched CUDA earlier in the same run (e.g. it just
finished training a table on GPU before starting inference) — forking after
that is unsafe regardless of what the child pool itself does. On a CPU-only
machine, this is byte-for-byte the same `fork` behavior as before.

### Per-worker GPU pinning

Unlike training (one job = one fresh process), inference's pool workers are
long-lived — each one builds its model **once** and reuses it for every batch
it's handed. So GPU assignment has to happen once, at worker startup, pinned
for that worker's whole life — not per batch.

```python
@staticmethod
def worker_init(get_wrapper_func_from_main, threads_per_worker=None, gpu_indices=None):
    ...
    if gpu_indices:
        ordinal = mp.current_process()._identity[0] - 1   # 1-based -> 0-based
        gpu_index = gpu_indices[ordinal % len(gpu_indices)]
        get_wrapper_func_from_main = functools.partial(get_wrapper_func_from_main, gpu_index=gpu_index)
    vae_model = get_wrapper_func_from_main()
```

`mp.current_process()._identity[0]` is private API — CPython's own stable,
long-standing 1-based pool-worker ordinal, used here because `mp.Pool`
doesn't hand workers any other way to distinguish themselves (every worker's
`initializer`/`initargs` are identical). This is called out explicitly rather
than hidden, and backed by a test that exercises a *real* `spawn` pool
(`test_pool_worker_ordinal_is_stable_1_based`) to catch the day this
assumption stops holding, without needing GPU hardware to do so.

**Deliberately not capped:** the number of worker processes (`n_jobs`, driven
by CPU thread budget, unrelated to GPU count) can exceed
`cuda_device_count()` — multiple workers then share one GPU
(`gpu_indices[ordinal % len(gpu_indices)]` wraps around). This is intentional:
the CVAE is small enough that sharing a GPU across a couple of workers is
cheap, and capping `n_jobs` by GPU count would leave CPU-bound work
(long-text KDE generation, etc.) under-parallelized for no good reason.

**Example — what actually happens for a user:**

```bash
infer --table_name customers --size 50000 --run_parallel true --batch_size 500
```

`size / batch_size = 100` batches. `n_jobs` (CPU-thread-budget-driven, say 8
on this box) workers get spun up under `spawn`. Each worker is pinned to one
of the 4 visible GPUs at startup (workers 0,4 → GPU 0; workers 1,5 → GPU 1;
etc.). All 100 batches get processed across those 8 GPU-backed workers
concurrently, instead of one CPU process working through them sequentially.

### Real bug #5 found post-merge: a merge from `main` silently dropped the `device` parameter

Unlike bugs #1-4, this one had nothing to do with GPU hardware specifics — it
broke **every non-parallel `infer` call, on every device, CPU included** —
and was only caught by running a basic train+infer smoke test after merging
`main` into this branch:

```
NameError: name 'device' is not defined
```

raised from inside `VaeInferHandler._get_wrapper`, at the line building the
wrapper's kwargs dict (`"device": device`). The method's own signature had no
`device` parameter to reference.

**Root cause:** this Phase 3 work (on this branch) and an unrelated PR on
`main` (a long-text NULL/empty-string fix) both independently touched code
immediately next to `_get_wrapper`, branching from the same parent commit.
This branch's version added a `device: Optional[torch.device] = None`
parameter to `_get_wrapper`; the `main` PR added a new helper method placed
directly above `_get_wrapper` and reworked the long-text generation code
around it, based on a copy of the file that predated the `device` parameter.
When `main` was later merged into this branch, the merge kept `main`'s
version of `_get_wrapper`'s signature (silently reverting the `device`
parameter) while correctly keeping both sides' other changes — a case of a
merge quietly discarding one side's edit to a function both branches
happened to touch nearby, rather than a mistake in either individual commit.

**Fix:** re-added the `device: Optional[torch.device] = None` parameter to
`_get_wrapper`. No other change needed — the call sites that pass
`device=resolve_device(gpu_index)` already existed and now resolve correctly
again.

**Regression test:** a new test asserts `_get_wrapper` accepts and forwards
`device` both as its default (`None`) and as an explicit value, by mocking
the wrapper-construction call and inspecting what it was invoked with.

**Lesson for future merges:** this class of bug — a real function signature
silently reverted by a merge that touched adjacent code — is invisible to
code review unless the diff against *both* parent branches is checked, and
is only reliably caught by actually running train/infer end-to-end after any
merge from `main`, not by unit tests alone (CPU-only unit tests happened to
still pass, since the missing parameter only breaks the one real call path
that supplies it). The recommended check this motivated: before opening a PR
that merges `main`, run `train` then `infer` for one small single-table and
one small related-table dataset, across every device configuration available
(CPU, single-GPU, multi-GPU if visible), with `run_parallel` both `true` and
`false` for infer. This exact shape of check — small data, every device, both
`run_parallel` states — is what would have caught this bug immediately; it
takes well under a minute of actual compute.

### Files touched (Phase 3)

- `src/syngen/ml/handlers/handlers.py` — `_select_mp_start_method`, `worker_init`, `_initialize_worker_vae_model`, `_get_wrapper` (new `device` param, later dropped by a merge from `main` and restored — see "Real bug #5" above).

---

## Real validation performed (not simulated)

This sandbox has 4x NVIDIA L4 GPUs — every claim below was actually run, not
inferred from code reading.

| Scenario | Dataset | Result |
|---|---|---|
| Single-GPU train+infer, mixed types, no text | `small_single_table` (5K rows) | Completed. Median accuracy 0.9643 |
| Single-GPU train+infer, real LSTM/text path | `long_text_table` (20K rows) | Completed after Phase 1 fixes. Median accuracy 0.9717 |
| Multi-GPU task-parallel training, 5 jobs across 4 GPUs (incl. PK+FK split) | relational set: `customers`/`products`/`orders`/`payments` | Confirmed via `ps` that 4 spawned worker processes ran concurrently, `nvidia-smi` showed multiple GPUs with active memory during the run. All 5 jobs completed. Median accuracy 0.94-0.97 across tables |
| Multi-GPU task-parallel inference, 10 batches | `small_single_table`, `--run_parallel true --batch_size 500` | Completed, 10 independent "Loaded VAE state" log lines confirming 10 separate worker model instances |

Unit test suite: all new tests run on CPU-only CI (mocking
`torch.cuda.is_available`/`device_count`) except the ones that exercise a
real `spawn` `Pool` for picklability/ordinal checks, which need no GPU
hardware either. A further regression test was added afterward for Real bug
#5 above (current suite: 1465 tests collected).

---

## Performance: now measured (was previously an open question)

The original version of this section listed "real wall-clock speedup
numbers" as unverified. That has since been measured properly, across all
single-table datasets and a `{2,4,6,8,16} tables × {4,8,16,32} cols ×
{1000,2000} rows` related-table matrix, on CPU / single-GPU / multi-GPU. A
dedicated benchmarking report covers train and infer timings, per-dataset
winners, and a follow-up investigation into `run_parallel` infer performance
and its own fallback-verification checks. Headline findings:

- **Multi-GPU training is a clear, decisive win for multi-table runs** —
  the more tables (and GPUs), the bigger the margin (up to ~4.6x faster than
  CPU at the largest related-table set tested).
- **GPU is a net loss for single small tables** — CPU wins every single
  single-table case tested (train and infer), since the CVAE is too small to
  amortize CUDA overhead.
- **`run_parallel=true` infer is only worth it on CPU** — on GPU, its
  `spawn`-per-worker startup cost (~100s fixed overhead) dominates at every
  scale tested (up to 256,000 rows), so `run_parallel=true` never wins there;
  on CPU it crosses over to a clear win as table size grows.

See that benchmarking report for full per-dataset tables and methodology.

## What still can't be verified without dedicated, repeated GPU access

(Carried over from the original plan — these weren't disproven, just not the
focus of validation so far.)

1. Multi-process GPU memory contention when `n_jobs` (inference workers) is
   much larger than `cuda_device_count()` under **sustained** load — the
   follow-up benchmarking above exercised large batch counts and large
   generated sizes without hitting OOM, but that's not the same as a
   long-running, sustained-contention test.
2. Driver-level races between sibling spawned processes concurrently
   initializing CUDA contexts under heavier concurrency than tested here.

## Where to look for more detail

- `tensorflow_to_pytorch.md` section 8.8, in this same directory — the
  durable summary of everything in this document, including the performance
  headline findings above, and the section that correctly predicted Phase 0's
  bug ahead of time but, lacking GPU hardware, could not have predicted the
  bugs found in Phases 1-3.
