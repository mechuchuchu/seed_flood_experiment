# CIFAR-100 SeedFlood Experiments

`cifar_seedflood.py` is a self-contained experiment runner for comparing ordinary
first-order training with communication-efficient, shared-seed gradient estimators on
CIFAR-100. It includes a CIFAR-style ResNet implementation, in-memory data loading,
five training modes, numerical diagnostics for zeroth-order optimization, YAML
configuration, and crash-safe JSON logging.

The central question is how much optimization quality is lost when workers communicate
only random seeds and scalar directional information instead of full model-sized
gradients.

## Contents

- [Experiment overview](#experiment-overview)
- [Training modes](#training-modes)
- [SeedFlood protocol](#seedflood-protocol)
- [Projected-gradient control](#projected-gradient-control)
- [Data pipeline](#data-pipeline)
- [Models, normalization, and activation functions](#models-normalization-and-activation-functions)
- [Parameter initialization](#parameter-initialization)
- [Optimizers and learning-rate schedules](#optimizers-and-learning-rate-schedules)
- [Numerical diagnostics](#numerical-diagnostics)
- [Performance and memory implementation](#performance-and-memory-implementation)
- [Installation and execution](#installation-and-execution)
- [Configuration files](#configuration-files)
- [Command-line reference](#command-line-reference)
- [Output format](#output-format)
- [Reproducibility](#reproducibility)
- [Practical guidance](#practical-guidance)
- [Limitations](#limitations)

## Experiment overview

The runner supports three estimator families:

| Family | Modes | Local computation per round | Communicated information |
|---|---|---:|---|
| First order | `fo` | One forward and backward pass | Not simulated |
| Zeroth order / SPSA | `zo_sign`, `zo_adam` | `2Q` forward passes per worker | `Q` shared, or `N*Q` per-node, seed/scalar pairs |
| Projected gradient | `proj_sign`, `proj_adam` | One backward pass plus `Q` projections per worker | `Q` shared, or `N*Q` per-node, seed/scalar pairs |

Here, `Q = n_queries`. The projected modes are a diagnostic control: they retain the
same seed-and-scalar communication representation as the ZO modes, but replace the
finite-difference directional derivative with an exact gradient projection. Comparing
the modes helps separate two effects:

1. `zo_*` versus `proj_*`: finite-difference bias and numerical cancellation.
2. `proj_*` versus `fo`: loss from reconstructing an update using a limited number of
   random directions.

This program simulates all workers in one process and on one device. It models the
information exchanged by SeedFlood; it does not create networked worker processes or
measure real network latency.

## Training modes

### `fo`

The baseline uses standard backpropagation with PyTorch SGD:

- momentum: `0.9`
- weight decay: `--weight_decay`
- default learning rate: `0.1`
- default schedule: cosine
- augmentation under `--augment auto`: enabled
- normalization under `--norm auto`: BatchNorm

One sampled batch is used per step. `n_rounds` therefore means optimizer steps in this
mode.

### `zo_sign`

Uses the shared-seed SPSA estimator described below, then applies a sign update:

```text
theta <- theta - learning_rate * sign(g_hat)
```

No momentum state is maintained.

### `zo_adam`

Uses the same SPSA estimator and applies an Adam-style update. The first- and
second-moment buffers are kept in FP32 for FP32/BF16 models and FP64 for FP64 models.
This is especially important for BF16, where coefficients close to one and small
moment increments may otherwise be rounded away.

### `proj_sign` and `proj_adam`

These modes obtain a true gradient with backward propagation, compress it into random
seeded projections, reconstruct a stochastic gradient estimate, and apply either the
sign or Adam update. They are experimental controls rather than pure zeroth-order
methods because each worker must be capable of backpropagation.

## SeedFlood protocol

Let `theta` denote all model parameters flattened into one vector. For a seed `s_q`,
every participant can regenerate the same Gaussian direction:

```text
z_q ~ Normal(0, I)
```

The random vector is never transmitted or permanently stored. Its seed is enough to
reproduce it in a deterministic sequence across all parameter tensors.

For zeroth-order training, each worker evaluates the same local batch at two symmetric
points:

```text
L_i+ = L_i(theta + mu * z_q)
L_i- = L_i(theta - mu * z_q)
```

In shared-seed mode, the server-side simulation averages worker losses and computes
the directional scalar:

```text
L_bar+ = mean_i(L_i+)
L_bar- = mean_i(L_i-)
s_q    = (L_bar+ - L_bar-) / (2 * mu)
```

The reconstructed estimator for `Q` queries is:

```text
g_hat = (1 / Q) * sum_q(s_q * z_q)
```

For isotropic Gaussian directions, this estimator is aligned with the gradient in
expectation. A finite `Q` leaves substantial direction-sampling variance, particularly
when the parameter dimension is much larger than `Q`.

### Shared seeds versus per-node seeds

`--seed_mode shared` is the primary SeedFlood setting:

- all nodes use the same `Q` directions;
- losses are averaged before each directional scalar is formed;
- the round contains `Q` reconstructed directions;
- batch noise is not reduced by increasing `Q`, because all queries reuse the same
  already-sampled node batches.

`--seed_mode per_node` gives every node its own directions:

- node `i` evaluates its own batch with its own seeds;
- the server averages `N * Q` terms of the form `s_iq * z_iq`;
- total direction and communication budgets differ from shared mode;
- each local gradient is transformed by a different random projection operator, so its
  finite-sample variance differs from the shared-direction estimator.

The training dataset is shuffled once with `data_seed`, divided into disjoint chunks,
and sampled independently within each chunk. Unless `gpu_resident` is enabled, these
partition indices remain on CPU with the source data.

### Why parameters are restored from a snapshot

ZO evaluation could move from `theta + mu*z` to `theta - mu*z` arithmetically and then
add `mu*z` again. The implementation deliberately does not do this. It clones `theta`
once per round and restores it with tensor copies after every evaluation. This avoids
accumulating rounding residuals, which are particularly damaging with BF16 parameters.

## Projected-gradient control

Projected modes first compute the average gradient `g` with backpropagation. For every
seeded direction they calculate:

```text
s_q = dot(g, z_q)
g_hat = (1 / Q) * sum_q(s_q * z_q)
```

This removes the finite-difference radius `mu`, curvature bias, and subtraction
cancellation from the measurement. `mu` may still appear in a configuration or output
filename, but it is not used by projected-gradient computation.

With independent Gaussian directions, reconstruction equals the true gradient in
expectation, not necessarily in an individual run—even when `Q` equals the parameter
dimension. Exact finite-sample reconstruction would require a complete appropriately
normalized orthogonal basis, which this implementation does not generate.

On evaluation rounds in shared mode, projected training also reports the cosine
similarity between `g_hat` and the true gradient.

## Data pipeline

The dataset is loaded from Hugging Face as `uoft-cs/cifar100`. This avoids dependence
on the torchvision dataset download service.

The full train and test images are converted to contiguous `uint8` tensors with shape
`(N, 3, 32, 32)`:

- training images: approximately 154 MB;
- test images: approximately 31 MB;
- labels: `int64` tensors.

By default, data remains in system RAM and selected batches are copied to the compute
device. `--gpu_resident` stores all four tensors in VRAM and removes repeated host-to-
device copies, at the cost of roughly 185 MB plus labels and allocator overhead.

Normalization uses the following CIFAR-100 statistics:

```text
mean = [0.5071, 0.4865, 0.4409]
std  = [0.2673, 0.2564, 0.2762]
```

Normalization is performed in FP32. Device copies of the mean and standard deviation
are cached. A batch is cast to the selected model dtype only after normalization and
augmentation.

### Augmentation

The optional augmentation is vectorized on the batch device:

1. reflection padding by four pixels;
2. independent random `32 x 32` crops;
3. independent horizontal flips with probability `0.5`.

Under `--augment auto`, augmentation is enabled only for `fo`. It is disabled for ZO
and projected modes so that stochastic augmentation does not contaminate paired loss
measurements. When explicitly enabled for ZO, augmentation is sampled once while the
batch is built, so the resulting transformed tensors are still identical for the plus
and minus evaluations of that round.

The default `fine_label` task has 100 classes. `--label_key coarse_label` switches to
the 20 superclass labels and adjusts the classifier output dimension accordingly.

## Models, normalization, and activation functions

Both supplied networks use a CIFAR-style stem: a `3 x 3`, stride-one convolution with
no ImageNet-style max pooling. Residual stages downsample using stride-two convolutions,
and the classifier follows adaptive global average pooling.

| Model | Blocks per stage | Widths | Approximate parameters on CIFAR-100 |
|---|---|---|---:|
| `resnet20` | `[3, 3, 3]` | `[16, 32, 64]` | 0.28 M |
| `resnet18` | `[2, 2, 2, 2]` | `[64, 128, 256, 512]` | 11.2 M |

`resnet20` is the practical default for expensive multi-query ZO experiments.

### Normalization

- `batch`: `BatchNorm2d`
- `group`: `GroupNorm` with eight groups
- `auto`: BatchNorm for `fo`, GroupNorm for `zo_*` and `proj_*`

BatchNorm is risky for symmetric ZO measurements because both the plus and minus
forward passes update running statistics. This introduces order-dependent state changes
between the two losses. The program warns if a ZO mode is explicitly combined with
BatchNorm.

### Activations

Supported values are `relu`, `gelu`, `silu`, `softplus`, `elu`, `leaky_relu`, and
`tanh`. `softplus` uses `beta=5.0`.

Smooth activations can be useful for ZO experiments. A perturbation may move ReLU units
across zero, placing the plus and minus evaluations in different linear regions. In
that case the finite difference captures transitions between regions rather than only
a local smooth directional derivative.

## Parameter initialization

`--init` supports:

- `default`: preserve the standard PyTorch module initialization;
- `kaiming_normal` and `kaiming_uniform`;
- `xavier_normal` and `xavier_uniform`;
- `orthogonal`.

For non-default schemes, convolution and linear biases are zeroed and affine
normalization parameters are reset to weight one and bias zero. Additional controls:

- `--init_gain`: multiply every convolution and linear weight by a global factor;
- `--fc_scale`: apply an additional factor to classifier weights and bias;
- `--zero_init_residual`: initialize the second normalization scale in every residual
  block to zero, so each block begins close to an identity mapping.

At startup the program reports the parameter count, `||theta||`, and `sqrt(d)`. For ZO
it also prints the approximate perturbation norm `mu * sqrt(d)` and its ratio to
`||theta||`. A large ratio is a warning that the finite-difference probe may no longer
be local.

## Optimizers and learning-rate schedules

The learning rate is evaluated once per one-indexed training round.

### Warmup

With `--warmup_rounds W`, the learning rate increases linearly:

```text
lr(round) = base_lr * round / W, for round <= W
```

### Constant and cosine schedules

After warmup, `constant` retains the base learning rate. `cosine` decays it according
to a half cosine over the remaining rounds. `auto` selects cosine for `fo` and constant
for all distributed-estimator modes.

### Weight decay

- `--weight_decay` is passed to SGD in `fo` and FO warmup.
- `--zo_weight_decay` is decoupled: before a ZO/projected update,
  `theta <- theta * (1 - lr * decay)`.

The latter does not enter the pseudo-gradient or contaminate Adam moment statistics.

### FO warmup diagnostic

`--fo_warmup_steps` optionally performs ordinary SGD before ZO or projected training.
This tests whether an estimator can continue learning after first-order optimization
escapes the near-uniform initial regime. It violates the intended seed-and-scalar
communication constraint and should be treated as a diagnostic, not a deployment
method. ZO Adam moments start from zero after the handoff.

## Numerical diagnostics

Diagnostics run only on evaluation rounds and can add significant computation.

### Curvature check

Enable with `--curvature_check`. In addition to `L+` and `L-`, the runner evaluates
`L0 = L(theta)` and records:

```text
dplus  = L+ - L0
dminus = L0 - L-
curv   = dplus - dminus = L+ + L- - 2*L0
```

For small `mu`, `curv` approximates `mu^2 * z^T H z`. Large asymmetry indicates that
the linear finite-difference approximation is poor.

### FP64 precision probe

Enable with `--precision_probe` in an FP32 or BF16 ZO run. The program temporarily
converts the model and probe batches to FP64, repeats the same seeded measurements,
then restores the original dtype and exact parameter snapshot. It records:

- `zo_delta`: the training-dtype value of `L+ - L-`;
- `zo_delta_fp64`: the FP64 reference;
- `zo_delta_err`: their difference;
- `zo_curv_fp64`: the FP64 curvature measurement.

The probe is disabled automatically for FP64 runs and non-ZO modes.

### Third-order bias probe

`--bias3_probe` implies `--precision_probe` for a ZO run. It measures symmetric
differences at both `mu` and `mu/2`. With

```text
Delta(mu) = L(theta + mu*z) - L(theta - mu*z),
```

the estimated cubic contribution is:

```text
bias3     = (Delta(mu) - 2*Delta(mu/2)) * 4/3
grad_term = Delta(mu) - bias3
```

The ratio `abs(zo_bias3 / zo_delta_fp64)` is a useful indication of how much of the
measured difference comes from third-order effects rather than the desired linear term.

### Collapse detection

`--early_abort` watches the median magnitude of the estimated directional signal
`|g^T z|`. The initial baseline is the median of the first three evaluation windows.
After `abort_min_round`, a run is stopped when both conditions hold:

1. the recent median signal is smaller than `baseline / abort_ratio`;
2. the best test loss remains at least `0.99 * log(number_of_classes)`.

The abort reason is written to the final JSON record.

### Multi-query and projected diagnostics

For more than one reconstructed direction, evaluation entries include scalar mean,
standard deviation, and direction count. Shared projected runs additionally include
the reconstructed-to-true-gradient cosine and true gradient norm.

## Performance and memory implementation

Several implementation details are designed specifically for high-query experiments:

- The dataset is stored as `uint8` rather than expanded float tensors.
- Batches are sampled directly from tensors; no `DataLoader` or worker processes are
  involved.
- All directions for a round are stored in one reusable contiguous matrix `Z` with
  shape `(R, d)`, where `d` is the parameter count and `R` is `Q` in shared mode or
  `N*Q` in per-node mode.
- Each row is filled parameter-by-parameter from its seed. This preserves the seeded
  parameter iteration order while still exposing the whole direction as a flat vector.
- Projected directional scalars are computed together with `Z @ g`, rather than with
  one dot-product reduction per query and parameter tensor.
- The final pseudo-gradient is reconstructed with one `Z.T @ scalars / R` matrix-vector
  operation. The flat result is split into zero-copy parameter-shaped views for the
  foreach optimizer operations.
- PyTorch foreach operations apply perturbations, copies, moment updates, and parameter
  updates with fewer kernel launches.
- Device scalar extraction is deferred: node losses and gradient losses are accumulated
  on device before conversion to Python values where possible.
- Seeds are generated in batches instead of one scalar extraction per query.
- FP32 master moment buffers are used for BF16 parameters.

The direction matrix intentionally trades memory for fewer kernel launches and no
second seed-replay pass during reconstruction. Its storage is:

```text
direction_bytes = number_of_directions * parameter_count * element_size
number_of_directions = Q       (shared)
                     = N * Q   (per-node)
```

For FP32 ResNet20 (`d` approximately 278K), the matrix is about 1.1 MB per direction:

| Directions | Approximate matrix memory |
|---:|---:|
| 1 | 1.1 MB |
| 16 | 17.8 MB |
| 64 | 71.2 MB |
| 256 | 285 MB |

For FP32 ResNet18 (`d` approximately 11.2M), 64 directions require about 2.9 GB. In
per-node mode, use `N*Q` in this calculation. Large-model or very-high-query runs can
therefore run out of memory and should reduce `Q`, reduce `N`, or use ResNet20.

Other main ZO memory costs are one exact parameter snapshot, one flat pseudo-gradient,
and—in Adam mode—two moment buffers. BF16 model parameters still use FP32 directions,
pseudo-gradients, and moments, so the direction matrix does not become smaller in a
BF16 run.

Compute cost scales differently by mode:

```text
ZO shared:        2 * Q * N local forward evaluations
ZO per-node:      2 * Q * N local forward evaluations
Projected shared: N local forward/backward evaluations + Q projections
Projected node:   N local forward/backward evaluations + N * Q projections
```

Here `N` is `n_nodes`. In this single-process simulator, local worker computations run
sequentially rather than concurrently.

## Installation and execution

The expected environment has Python, PyTorch with a compatible CUDA build, NumPy,
Hugging Face Datasets, and PyYAML. In the supplied Vast PyTorch image:

```bash
source /venv/main/bin/activate
uv pip install datasets pyyaml
```

From the repository root:

```bash
cd /workspace/seed_flood_experiment
python resnet/cifar_seedflood.py --help
```

The first dataset load downloads CIFAR-100 into the Hugging Face cache. Later runs use
the cached copy.

### First-order baseline

```bash
python resnet/cifar_seedflood.py \
  --mode fo \
  --model resnet20 \
  --n_rounds 3000 \
  --eval_every 200 \
  --torch_seed 0
```

### Shared-seed ZO Adam

```bash
python resnet/cifar_seedflood.py \
  --mode zo_adam \
  --model resnet20 \
  --norm group \
  --lr 3e-4 \
  --mu 2.5e-3 \
  --beta1 0.999 \
  --beta2 0.9999 \
  --n_queries 4 \
  --n_nodes 4 \
  --seed_mode shared \
  --n_rounds 3000 \
  --warmup_rounds 200 \
  --gpu_resident \
  --torch_seed 0
```

### Projected-gradient comparison

```bash
python resnet/cifar_seedflood.py \
  --mode proj_adam \
  --model resnet20 \
  --norm group \
  --lr 3e-4 \
  --n_queries 64 \
  --n_nodes 1 \
  --seed_mode shared \
  --n_rounds 3000 \
  --torch_seed 0
```

### Precision and bias study

```bash
python resnet/cifar_seedflood.py \
  --mode zo_adam \
  --dtype fp32 \
  --mu 1e-3 \
  --precision_probe \
  --curvature_check \
  --bias3_probe \
  --eval_every 200
```

These probes perform extra forward passes and FP64 model conversions. Use a reasonably
large `eval_every` for long sweeps.

## Configuration files

Pass YAML defaults with `--config`:

```bash
python resnet/cifar_seedflood.py --config resnet/zo_base_1.yaml
```

CLI arguments override YAML values:

```bash
python resnet/cifar_seedflood.py \
  --config resnet/zo_base_1.yaml \
  --mu 1e-3 \
  --lr 1e-4 \
  --out resnet/results/custom_run.json
```

Every top-level YAML key must exactly match an argparse destination. Unknown keys cause
an immediate error, which prevents silent configuration typos. Values loaded from YAML
are not separately type-coerced by argparse, so use proper YAML numeric and boolean
types.

Included examples:

- `base_zo.yaml`: compact ZO Adam configuration with all numerical probes;
- `zo_base_1.yaml`: long SiLU ZO Adam experiment;
- `zo_base_2.yaml`: ReLU experiment with a short FO warmup;
- `softplus_test.yaml`: long smooth-activation experiment;
- `proj_test.yaml`: projected-gradient comparison configuration.

Paths are resolved relative to the current shell directory. The automatic output path
is `results/...`, so running from the repository root writes to the root-level
`results/`, while running from `resnet/` writes to `resnet/results/`. Set `--out`
explicitly when a stable location matters.

## Command-line reference

### Core mode and model options

| Option | Default | Description |
|---|---:|---|
| `--config PATH` | none | YAML file supplying parser defaults. |
| `--mode` | `fo` | `fo`, `zo_sign`, `zo_adam`, `proj_sign`, or `proj_adam`. |
| `--model` | `resnet20` | `resnet20` or `resnet18`. |
| `--norm` | `auto` | `auto`, `batch`, or `group`. |
| `--dtype` | `fp32` | Model/forward dtype: `fp32`, `bf16`, or `fp64`. |
| `--act` | `relu` | Activation used throughout the network. |
| `--label_key` | `fine_label` | `fine_label` for 100 classes or `coarse_label` for 20. |

### Training options

| Option | Default | Description |
|---|---:|---|
| `--lr` | mode-dependent | `0.1` for FO, `1e-3` otherwise. |
| `--warmup_rounds` | `0` | Number of linear LR warmup rounds. |
| `--lr_schedule` | `auto` | `auto`, `constant`, or `cosine`. |
| `--batch_size` | `128` | Batch size sampled independently for each node. |
| `--n_rounds` | `3000` | Number of optimizer rounds/steps. |
| `--eval_every` | `200` | Evaluation and logging interval. |
| `--weight_decay` | `5e-4` | SGD weight decay for FO and FO warmup. |
| `--augment` | `auto` | `auto`, `on`, or `off`. |

The effective total batch examples sampled per distributed round are
`batch_size * n_nodes`.

### ZO and projected-estimator options

| Option | Default | Description |
|---|---:|---|
| `--mu` | `1e-2` | Symmetric finite-difference radius; ZO only. |
| `--beta1` | `0.99` | Adam first-moment coefficient. |
| `--beta2` | `0.999` | Adam second-moment coefficient. |
| `--eps` | `1e-8` | Adam denominator epsilon. |
| `--zo_weight_decay` | `0` | Decoupled weight decay for ZO/projected modes. |
| `--seed_mode` | `shared` | `shared` or `per_node`. |
| `--n_queries` | `1` | Directions per round and per node where applicable. |
| `--n_nodes` | `1` | Number of simulated worker partitions. |

### Initialization options

| Option | Default | Description |
|---|---:|---|
| `--init` | `default` | Weight initialization scheme. |
| `--init_gain` | `1.0` | Global convolution/linear weight multiplier. |
| `--fc_scale` | `1.0` | Additional final classifier multiplier. |
| `--zero_init_residual` | false | Start residual branches with zero second norm scale. |

### Diagnostic options

| Option | Default | Description |
|---|---:|---|
| `--precision_probe` | false | Repeat selected ZO measurements in FP64. |
| `--bias3_probe` | false | Estimate cubic finite-difference bias; enables precision probing for ZO. |
| `--curvature_check` | false | Evaluate unperturbed loss and measure plus/minus asymmetry. |
| `--fo_warmup_steps` | `0` | Diagnostic SGD steps before ZO/projected training. |
| `--fo_warmup_lr` | `0.05` | Learning rate during FO warmup. |
| `--early_abort` | false | Stop collapsed ZO runs early. |
| `--abort_min_round` | `1000` | Earliest round eligible for collapse termination. |
| `--abort_ratio` | `100` | Required signal reduction factor. |

### Runtime and output options

| Option | Default | Description |
|---|---:|---|
| `--gpu_resident` | false | Keep the complete dataset on the GPU. |
| `--data_seed` | `0` | Seed controlling the disjoint node partition. |
| `--torch_seed` | none | Seed for model initialization and subsequent global PyTorch randomness. |
| `--out PATH` | automatic | JSON result path. |

## Output format

Results are written as JSON after every evaluation using a temporary file followed by
an atomic replacement. A process failure therefore normally leaves the most recent
complete evaluation record rather than a partially written JSON document.

The top-level structure is:

```json
{
  "args": {},
  "n_params": 278324,
  "started_at": "2026-08-29 10:43:31",
  "history": [],
  "final": {},
  "init_stats": {},
  "rounds_done": 3000
}
```

Each history entry always contains:

| Field | Meaning |
|---|---|
| `round` | Completed training round. |
| `train_avg` | Mean sampled training loss since the previous evaluation. |
| `test_loss` | Full test-set cross-entropy. |
| `test_acc` | Full test-set accuracy as a fraction. |
| `lr` | Learning rate used in that round. |
| `elapsed` | Seconds elapsed since the current training phase began. |

Depending on mode and diagnostics, entries may additionally contain:

| Field | Meaning |
|---|---|
| `zo_scalar_mean`, `zo_scalar_std` | Directional-scalar distribution. |
| `zo_n_directions` | Number of reconstructed directions. |
| `zo_signal_med` | Median absolute scalar over the logging window. |
| `zo_signal_zero_frac` | Fraction of exactly zero signals in the window. |
| `zo_dplus`, `zo_dminus`, `zo_curv` | Training-dtype curvature diagnostics. |
| `zo_delta`, `zo_delta_fp64`, `zo_delta_err` | Precision comparison. |
| `zo_curv_fp64` | FP64 curvature reference. |
| `zo_bias3`, `zo_grad_term` | Third-order bias estimate and corrected linear term. |
| `proj_cosine` | Cosine similarity between reconstructed and true gradients. |
| `proj_grad_norm` | Norm of the true projected-mode gradient. |

`final` contains final test metrics, total seconds, and an optional early-abort reason.
If FO warmup is used, a separate `fo_warmup` object records its handoff metrics.

Automatic filenames encode the principal mode, model, dtype, optimizer parameters,
batch/node/round counts, and selected non-default settings. They are convenient for
aggregation but should not replace the full `args` object as the authoritative record.

## Reproducibility

Use both seeds for repeatable experiments:

```bash
--torch_seed 0 --data_seed 0
```

`torch_seed` is applied before model construction and governs the global PyTorch RNG
used for initialization, batch sampling, augmentation, and query seed generation.
`data_seed` independently governs the one-time partition of examples among nodes.

The random direction implementation intentionally fills each flat row in a fixed
parameter iteration order and uses FP32 for FP32/BF16 models and FP64 for FP64 models.
Within a round, the same stored row is used for perturbation, projection, precision
probing, and pseudo-gradient reconstruction. Its seed is still sufficient to reproduce
the row on another participant.

Exact cross-machine equality is not guaranteed. CUDA kernels, library versions,
floating-point reduction order, and nondeterministic backend algorithms may differ.
The program does not enable `torch.use_deterministic_algorithms` or configure cuDNN
determinism.

## Practical guidance

### Choosing `mu`

`mu` balances two failure modes:

- too large: curvature and higher-order terms bias the directional derivative;
- too small: `L+ - L-` is lost to floating-point cancellation or model-dtype rounding.

Monitor `zo_curv`, `zo_delta_err`, and `zo_bias3` rather than selecting `mu` solely by
parameter scale. The startup ratio `mu*sqrt(d) / ||theta||` is a useful first check.

### Choosing `n_queries`

Larger `Q` reduces random-direction variance but increases computation and seed/scalar
communication linearly. It does not reduce batch noise when the same sampled batches
are reused across queries. Increasing batch size or node count addresses a different
noise source.

### Choosing dtype

- FP32 is the normal experimental default.
- BF16 reduces parameter/forward memory but can erase small symmetric loss differences;
  use the precision probe to quantify this.
- FP64 is useful as a diagnostic reference but is significantly slower and consumes
  more memory.

### Comparing modes fairly

For a controlled comparison, keep the model, normalization, activation, initialization,
batch partitions, seeds, query count, optimizer family, learning-rate schedule, and
augmentation policy fixed. `auto` selects different normalization and augmentation for
FO, so explicitly set these flags when architectural identity matters.

### Monitoring uniform collapse

For 100 classes, random uniform predictions have cross-entropy `log(100) ~= 4.605` and
accuracy near `0.01`. For 20 coarse classes the corresponding loss is
`log(20) ~= 2.996` and accuracy is near `0.05`. Persistent values around these levels,
combined with a collapsing directional signal, indicate a failed run.

## Limitations

- Workers and communication are simulated sequentially in one Python process.
- There is no real distributed backend, network serialization, straggler behavior, or
  communication timing.
- Every node uses the same `batch_size`; heterogeneous data volume and compute are not
  modeled.
- Node partitions are IID after a global random permutation; label-skewed non-IID
  federated partitions are not implemented.
- Gaussian directions are dense, so materialization, storage, and projection still
  scale with model size even though communication does not.
- The flat `(N*Q, d)` direction matrix can be prohibitively large for ResNet18 or
  high-query per-node experiments; there is currently no streaming fallback.
- The full dataset must fit in host RAM, or VRAM with `--gpu_resident`.
- BatchNorm running-state behavior makes it unsuitable for clean ZO comparisons.
- Precision probing repeatedly converts the entire model between dtypes and is intended
  for diagnostics, not throughput benchmarks.
- Automatic output paths depend on the current working directory.
