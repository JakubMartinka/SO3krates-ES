# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository layout

The working directory (`so3krates/`) is not itself a git repo. The actual code lives in the
**`mlff/`** subdirectory, which is a clone of https://github.com/thorben-frank/mlff — this is
where the git history, `.github/workflows/`, and all commands below apply. All paths in this
file are relative to `mlff/` unless stated otherwise.

`pc_RE_dgrad_5950_training_db.json` at the repo root is a large (~280MB) training database
loadable with `mlatom`; it is data, not part of the `mlff` codebase.

## What this is

`mlff` is a JAX/Flax library for training, evaluating, and running MD with machine-learned force
fields, centered on the **SO3krates** equivariant transformer (see `mlff/README.md` for the papers
and the full CLI walkthrough). Training/eval/MD are primarily driven through console-script
entry points, not a typical `import`-and-run library, though the model and potential classes are
fully usable from Python (see README's "Use `So3krates` in Python" section).

## Commands

Run all commands from within `mlff/` (the package repo root).

```bash
# Install (editable). Install jax/jaxlib matching your CUDA version FIRST — see mlff/README.md.
pip install -e .

# Run the full test suite
pytest tests/

# Run a single test file / single test
pytest tests/test_so3krates.py
pytest tests/test_so3krates.py::test_so3krates_init

# Train a SO3krates model from an ASE-readable file or .npz
train_so3krates --data_file data.xyz --n_train 1000 --n_valid 100 --ckpt_dir first_module

# Evaluate a trained checkpoint (run from inside the ckpt dir, or point --ckpt_dir at it)
cd first_module && evaluate

# MD (mdx path — see architecture notes below); requires the external `glp` package
run_relaxation --qn_max_steps 1000 --qn_tol 0.0001 --use_mdx
trajectory_to_xyz --trajectory relaxed_structure.h5 --output relaxed_structure.xyz
run_md --start_geometry relaxed_structure.xyz --thermostat velocity_verlet \
       --temperature_init 600 --time_step 0.5 --total_time 1 --use_mdx
```

Other console scripts (all defined in `setup.py`'s `entry_points`, implemented in `mlff/cAPI/`):
`train`, `run_md`, `run_relaxation`, `analyse_md`, `train_so3kratACE`
(`train_so3kratace`), `trajectory_to_xyz`, `to_mlff_input`.

CI (`.github/workflows/CI.yml`) runs on Python 3.9 with `pip install jax jaxlib tensorflow` then
`pip install -e .` and `pytest tests/`.

By default `train_so3krates`/`train`/etc. call `wandb.init(...)` unconditionally
(`--use_wandb` defaults to `True`, and it's an argparse `type=bool` flag so passing
`--use_wandb False` does **not** disable it — any non-empty string is truthy). If there's no
`wandb` account configured, either run `wandb login` once, or set `WANDB_MODE=offline` (or
`disabled`) in the environment before invoking any training command.

### Local dev environment

`setup.py` leaves `jax`/`jaxlib`/`flax`/`optax` unpinned (only `orbax-checkpoint==0.5.23` is
pinned). A plain `pip install -e .` today pulls the latest `jax`/`flax`, which breaks this
2023-era codebase (`jax.tree_map` was removed, `jnp.array(None)` semantics changed) — 8 of the 28
tests fail with `AttributeError`/`ValueError` under unpinned installs. The versions verified to
pass the full suite are:

```bash
python3.11 -m venv .venv   # inside mlff/
.venv/bin/pip install -e .
.venv/bin/pip install "jax==0.4.28" "jaxlib==0.4.28" "flax==0.8.5" "optax==0.2.2"
.venv/bin/python -m pytest tests/   # 28 passed
```

A local `.venv/` under `mlff/` is the expected place to develop; it is not committed.

## Architecture

### Package map (`mlff/mlff/`)

- **`nn/`** — the model itself (Flax `linen` modules): `StackNet` assembly, layers, embeddings,
  observables. Go here to add an architecture, layer type, or output quantity.
- **`training/`** — the training loop: `Coach` (orchestration/config), `run.py` (jitted
  train/valid step loop + checkpointing), `loss.py`, `optimizer.py`, `train_state.py`, plus
  `lr_decay/` and `stopping_criteria/`.
- **`data/`** — dataset abstraction (`data.py::DataTuple`, `dataset.py::DataSet`), ASE/npz
  loaders (`dataloader.py`, `dataloader_new.py::AseDataLoader`), `preprocessing.py`. Converts raw
  structures into padded input dicts.
- **`geometric/`** — plain-array (non-Flax) geometry math: distance matrices, minimum-image
  convention, rotations. Used by data preprocessing and tests, not inside the Flax graph.
- **`sph_ops/`** — spherical-harmonics / Clebsch-Gordan machinery (`base.py`, `contract.py`); the
  equivariant math backbone used by `nn/embed` and `nn/layer/so3krates_layer.py`.
- **`cAPI/`** — the CLI entry points (`mlff_train_so3krates.py`, `mlff_eval.py`, `mlff_md.py`,
  etc.) — where CLI args get turned into `nn`/`training`/`data` objects.
- **`io/`** — checkpoint I/O and JSON hyperparameter save/load
  (`load_params_from_ckpt_dir`, `read_json`, `save_dict`).
- **`indexing/`** — builds neighbor-list index arrays (`idx_i`/`idx_j`), padded to a fixed
  `n_pairs_max`, with MIC support.
- **`padding/`** — computes/applies fixed-shape padding for positions/types/indices so batches
  have static shapes for `jax.jit`.
- **`masking/`** — `safe_mask`/`safe_scale`: zero out padded atoms/pairs without producing NaN
  gradients. Used pervasively wherever padding is present.
- **`properties/`** — `property_names.py` defines canonical property-name constants (`pn.energy`,
  `pn.force`, ...); `properties.py` maps them to dataset-specific keys (e.g.
  `md17_property_keys`). See the cross-cutting note below.
- **`cutoff_function/`**, **`basis_function/`** — cutoff functions and radial-basis/spherical-
  harmonic expansions used by the geometry embedding.
- **`random/`** — RNG splitting/seeding helpers.

### Model assembly (`nn/stacknet`, `nn/layer`, `nn/base`)

`nn/stacknet/stacknet.py::StackNet` is the top-level Flax module. It holds
`geometry_embeddings`, `feature_embeddings`, `layers`, `observables` (each a sequence of
callables) plus `prop_keys`. Its `__call__`:

1. builds `point_mask`/`pair_mask` (from `z != 0` and `idx_i != -1` padding sentinels),
2. runs geometry embeddings (e.g. `GeometryEmbed`) to get `rbf_ij`, `sph_ij`, `chi`, `phi_r_cut`,
3. runs feature embeddings (atom-type embedding) into `x`,
4. loops `layers`, each consuming/updating a shared `quantities` dict — e.g.
   `So3kratesLayer.__call__(x, chi, rbf_ij, sph_ij, phi_r_cut, idx_i, idx_j, pair_mask,
   point_mask, ...)` returns `{'x': ..., 'chi': ...}` merged back in,
5. runs `observables` (e.g. `Energy`) against the final `quantities`.

`nn/layer/so3krates_layer.py::So3kratesLayer` is the core block: a `FeatureBlock` (invariant
attention over `x`) and a `GeometricBlock` (equivariant attention updating `chi` via
`make_l0_contraction_fn` from `sph_ops`), plus an `InteractionBlock` mixing invariant features
with SPHC norms. `nn/layer/schnet_layer.py` is an alternative invariant-only layer;
`nn/layer/h_register.py::get_layer(name, h)` is the factory used to reconstruct layers from a
JSON hyperparameter dict.

`nn/base/sub_module.py::BaseSubModule` is the common base for every embedding/layer/observable:
it enforces `__dict_repr__()` (a JSON-serializable self-description used for checkpoint
hyperparameter dumps and reconstruction) and `reset_prop_keys()`. This is the round-trip
mechanism (live model → `hyperparameters.json` → reconstructed model) used by
`MLFFPotential.create_from_ckpt_dir`.

`nn/representation/so3krates.py::So3krates` is the convenience constructor (used by
`mlff_train_so3krates.py`) that wires `GeometryEmbed` + atom embeddings + a stack of `L`
`So3kratesLayer`s + the given observables into a `StackNet`.

### Training pipeline

`cAPI/mlff_train_so3krates.py::train_so3krates()`: parse CLI args → load data (`.npz` or ASE via
`AseDataLoader`) → build `DataSet`, split, apply energy shift → build `mlff.nn.So3krates` and the
observable function (forces/stress derived via `jax.jacrev` autodiff over energy, then
`jax.vmap`'d over the batch, in `nn/stacknet/observable_function.py`) → build `Optimizer` and
`Coach` (`training/coach.py`) → build the loss (`training/loss.py`) → `DataTuple` filters the
data dict to the model's input/target keys → init or restore `TrainState` → `coach.run(...)` →
`training/run.py::run_training` (the actual loop).

JAX-specific patterns worth knowing before touching `training/run.py`:
- `train_step_fn`/`valid_step_fn` are `jax.jit`'d with the loss/metric fn as a static arg, using
  `jax.value_and_grad(loss_fn, has_aux=True)`.
- All batches are pre-padded to fixed atom/pair counts (via `padding`/`indexing`) so `jit` doesn't
  retrace per batch.
- NaN/Inf loss or grad-norm triggers a soft restart from the best orbax checkpoint
  (`restart_by_nan=True`).
- Checkpointing uses `orbax.checkpoint.CheckpointManager` with `AsyncCheckpointer` and
  best-metric tracking.
- `flax.linen.Module.sow('record', ...)` inside attention blocks stashes attention
  coefficients for later inspection; this collection gets reset on NaN restarts.

### `md/` vs `mdx/`

- **`md/`** is the older, self-contained MD stack.
- **`mdx/`** is the newer stack built on the external **`glp`** package
  (github.com/sirmarcel/glp), which supplies the neighbor-list/`Graph` abstraction. `mdx/atoms.py`
  imports `glp.system.unfold_system`, `glp.neighborlist.quadratic_neighbor_list`, etc., and raises
  an `ImportError` with an install hint if `glp` is missing.
  `mdx/potential/mlff_potential.py::MLFFPotential.create_from_ckpt_dir` rebuilds a `StackNet` from
  a checkpoint's `hyperparameters.json`, restores params, then flips
  `input_convention` → `'displacements'` and `output_convention` → `'per_atom'` (via the
  `reset_*` methods from `BaseSubModule`) before wrapping it as a `potential_fn(graph)` — this is
  how a trained energy/positions model becomes a per-atom/displacement-based potential for MD
  without touching any network code.
  **New MD/ASE-calculator work should go in `mdx/`, not `md/`** (the README notes MD is
  "currently under re-write" and points at `mdx`).

### Cross-cutting: `prop_keys` and mask-safe equivariance

Every model submodule takes a `prop_keys: Dict` field instead of hardcoding dict keys.
`properties/property_names.py` defines canonical symbolic names (`pn.energy`, `pn.force`,
`pn.atomic_type`, ...); `properties/properties.py` maps these to dataset-specific string keys
(e.g. `md17_property_keys`). Modules resolve `self.energy_key = self.prop_keys[pn.energy]` in
`setup()`, and `StackNet.reset_prop_keys()` / `reset_input_convention()` /
`reset_output_convention()` propagate changes recursively — this indirection is what lets the
exact same network be reused across differently-keyed datasets and across the training vs.
MD/mdx input/output conventions.

Separately, `masking/mask.py::safe_mask`/`safe_scale` are used everywhere padding exists
(`idx_i == -1` pairs, `z == 0` atoms) to zero contributions while keeping gradients finite.
Together with `sph_ops` (CG contractions), `geometric`/`indexing` (fixed-length neighbor lists),
and `segment_sum`-based aggregation in `so3krates_layer.py`, this is how the whole `StackNet`
stays `jax.jit`-compatible across a batch of differently-sized structures.

## Data file conventions

`--data_file` accepts anything `ase.io.read` can load, or `.npz` files. Default `.npz` key
mapping: `atomic_position` → `R`, `atomic_type` → `z`, `energy` → `E`, `force` → `F` (plus
`unit_cell`/`pbc` when `--mic` is used). Override via `--prop_keys` (see `mlff/README.md` for the
full deep-dive on units, energy shifts, and hyperparameter flags like `--L`/`--F`/`--degrees`/
`--r_cut`).

`z` may be given with shape `(n,)` instead of `(n_data, n)` if every structure shares the same
atom ordering/composition — `mlff` detects the missing leading axis and repeats it (see
`mlff/mlff/data/dataloader.py` warning "Detected missing data dimension (0-th axis) for z").

### Using mlatom-prepared data (e.g. `mlatom` `.json` molecular databases)

This workspace also has an `mlatom` checkout at `/home/jakub/UK/AV/PROJECTS/mlatom/mlatom` that
is **not** pip-installed into `mlff`'s venv. To `import mlatom` from any script/REPL, put its repo
root on `PYTHONPATH`:

```bash
export PYTHONPATH="/home/jakub/UK/AV/PROJECTS/mlatom/mlatom:$PYTHONPATH"
```

`mlatom.data.molecular_database.load(path, format='json')` gives molecules with `.energy` in
**Hartree** and per-atom `.energy_gradients` in **Hartree/Angstrom** (`.xyz_coordinates` are
already Angstrom). To train `mlff` on such data:

1. Convert to an `.npz` with the default keys — `R` (Angstrom), `z`, `E` (leave in Hartree),
   `F = -energy_gradients` (note the sign flip: mlff trains on forces, mlatom stores gradients).
   See `convert_100_to_npz.py` at the repo root for a worked example (100-geometry, single-
   molecule trajectory from `100.json` → `100_training_data.npz`).
2. Train with `--units energy=Hartree,force="Hartree/Angstrom"` so `mlff` rescales into its
   internal eV/Angstrom convention (`ase.units` supplies `Hartree`; `Angstrom == 1`, so the force
   factor is just the Hartree→eV constant).
