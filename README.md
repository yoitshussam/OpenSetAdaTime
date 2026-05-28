# OpenSet-AdaTime

Unified benchmark for **closed-set, OSDA, PDA and UniDA** domain adaptation on
time-series Human Activity Recognition (HAR). Combines re-implemented
scenario-native methods (OSBP, TSFA, SPADA, PDAAN) with seven UniDA methods
(UDA, OVANet, DANCE, PPOT, UniOT, UniJDOT, RAINCOAT) and eighteen closed-set
baselines under a single training/evaluation protocol, plus a hardest-first
curriculum sweep over private-class counts.

## 1. Requirements

The codebase is developed against the following pinned versions; matching them
exactly is the safest bet, anything reasonably close should also work.

| Package        | Version      |
|----------------|--------------|
| python         | 3.9          |
| torch          | 2.7.1+cu118  |
| torchvision    | 0.22.1+cu118 |
| torchmetrics   | 1.8.2        |
| numpy          | 2.0.2        |
| pandas         | 2.3.3        |
| scipy          | 1.13.1       |
| scikit-learn   | 1.6.1        |
| scikit-image   | 0.24.0       |
| matplotlib     | 3.9.4        |
| seaborn        | 0.13.2       |
| POT (`ot`)     | 0.9.6        |
| optuna         | 4.8.0        |
| mlflow         | 3.1.4        |
| tqdm           | 4.67.1       |


### Start an MLflow server first

Every entry-point (`main.py`, `main_sweep.py`, `run_curriculum.py`,
`extract_best_hparams.py`) logs to MLflow over HTTP at
`http://127.0.0.1:5001`. Launch the tracking server **before** training:

```bash
mlflow server --host 127.0.0.1 --port 5001
```

If you want a different port, change it both on the server command line and in
the `mlflow.set_tracking_uri(...)` call at the top of `main.py`,
`main_sweep.py`, `run_curriculum.py` and `extract_best_hparams.py`.

## 2. Datasets

Three public HAR datasets are used (we resample everything to 50 Hz and window
at 150 samples = 3 s):

| Dataset    | Subjects | Sensor location          | Notes                                                |
|------------|----------|--------------------------|------------------------------------------------------|
| RealWorld  | 15       | waist (accel + gyro)     | ~41,900 windows. Split into `_male` / `_female`      |
| Pamap2     |  9       | wrist + chest + ankle    | ~19,100 windows after dropping the "transient" class |
| MHEALTH    | 10       | chest + ankle + wrist    | ~4,600 windows                                       |

The four classes shared by all three datasets — **lying / sitting / walking /
running** — form the closed-set label space; every other class is treated as
private to its source or target in OSDA/PDA/UniDA.

### Preprocessing

The raw → windowed pipeline lives in `preprocessing/`:

- `preprocessing/raw_data_processing.py` — per-dataset parsers (downloads,
  resampling, segmentation).
- `preprocessing/run_datasets.py` — driver: builds `*_processed.pkl` files
  ready for the dataloader.
- `preprocessing/split_realworld_by_gender.py` — produces
  `RealWorld_male_processed.pkl` and `RealWorld_female_processed.pkl` for the
  RealWorld male→female protocol used in the hyper-parameter sweep.

Run them once to produce the `*_processed.pkl` files; everything downstream
expects them at `--data_path` (default `../dataset`).

## 3. Algorithms

All algorithms live in `algorithms/algorithms.py`. Each is a subclass of
`Algorithm` and declares which scenario(s) it covers via the `SCENARIO`
attribute (used by the `--scenario` shortcut in `main.py`).

| Family       | Methods                                                      |
|--------------|--------------------------------------------------------------|
| Closed-set   | NO_ADAPT, TARGET_ONLY, DANN, CDAN, DDC, Deep_Coral, DSAN, HoMM, MMDA, DIRT, AdvSKM, DAAN, CoDATS, CoTMix, CLUDA, SASA, SSSS_TSA, SWL_Adapt, ACON, uDAR |
| OSDA-native  | OSBP, TSFA                                                   |
| PDA-native   | SPADA, PDAAN                                                 |
| UniDA        | UDA, OVANet, DANCE, PPOT, UniOT, UniJDOT, RAINCOAT           |

Backbones live in `models/models.py`; default is `FNO` (Fourier Neural
Operator), `CNN` is also available via `--backbone CNN`.

## 4. Training procedure

The full pipeline is **sweep → extract → train → curriculum → plot**.

### 4.1 Train a single configuration

`main.py` runs one (source, target, algorithm, scenario) configuration with
`--num_runs` random seeds and logs everything to MLflow.

```bash
# One method on one pair:
python main.py \
    --source_dataset RealWorld \
    --target_dataset Pamap2 \
    --da_method UniJDOT \
    --scenario UniDA \
    --backbone FNO \
    --num_runs 5 \
    --exp_name EXP1

# Or run every method registered for a scenario:
python main.py --source_dataset RealWorld --target_dataset Pamap2 \
               --scenario OSDA --da_method ALL --num_runs 5
```

Hyper-parameters for the training run come from `configs/hparams.py`
(`get_hparams_class(source_dataset, backbone)`), which is the only place
`main.py` / `run_curriculum.py` look. `best_hparams.json` (see §4.2) is a
sweep artifact — you have to copy its values into `configs/hparams.py`
yourself for them to take effect.

### 4.2 Hyper-parameter sweep + extraction

We pick hyper-parameters once, on the **RealWorld male → female** within-dataset
split, then reuse them for every cross-dataset run. `main_sweep.py` runs Optuna
trials and `extract_best_hparams.py` distills the best trial into
`best_hparams.json`.

```bash
# Sweep one method (50 Bayesian trials, 3 seeds each):
python main_sweep.py \
    --source_dataset RealWorld_male \
    --target_dataset RealWorld_female \
    --scenario UniDA \
    --da_method UniJDOT \
    --num_runs 3 \
    --num_sweeps 50 \
    --hp_search_strategy bayes \
    --metric_to_minimize H_score \
    --exp_name sweep_unijdot

# Extract the best trial for that algorithm/scenario into best_hparams.json:
python extract_best_hparams.py \
    --exp_name sweep_unijdot \
    --source_dataset RealWorld_male \
    --target_dataset RealWorld_female
```

`best_hparams.json` is **not** read by `main.py` / `run_curriculum.py`; it is
a sweep artifact you inspect and then copy the relevant values into
`configs/hparams.py` (under `alg_hparams[<method>]` for the matching source
dataset class) before re-running.

### 4.3 Curriculum runs over private-class counts

`run_curriculum.py` adds the top-`n` hardest private classes (as ranked by FNO
mean-cosine distance to the source-known prototypes) and runs one training
experiment per `n`. One CLI invocation = one process, so multiple `n` values
can be fanned out by a shell loop.

```bash
# OSDA: 4 hardest target-privates added to the target side, 5 seeds.
python run_curriculum.py \
    --source_dataset RealWorld \
    --target_dataset Pamap2 \
    --scenario OSDA --strategy hard --n_unknown 4 \
    --da_method UniJDOT --backbone FNO --num_runs 5
```

Scenarios:
- `OSDA` — grows the **target** side by `n` target-private classes.
- `PDA`  — grows the **source** side by `n` source-private classes.
- `UniDA` — does both, with the same `n` per side.

Ranking JSONs are read from `feature_distance_4known_fno_mean/` (OSDA target
ranking) and `feature_distance_4known_fno_mean_pda/` (PDA source ranking).

The example shell wrappers under `bashes_and_logs/run_curriculum_*.sh` show
how a full sweep over all six pairs × all methods × all `n` is launched.

### 4.4 Aggregating + plotting results

Once the MLflow store has the runs, dump them to CSV and plot:

```bash
# Export every run's metrics + params into analysis/runs.csv.
python plotting/export_mlruns_csv.py

# Parse stdout training logs in experiments_logs/ into a per-step CSV
# (used for loss-curve plots).
python plotting/parse_training_logs.py
```

Visualizations (all write to `figures/`):

```bash
# H-score / OS* / UNK / F1 curves vs. n, per pair (2x3 grid):
python plotting/plot_curves.py --experiment OSDA --metric H_score
python plotting/plot_curves.py --experiment UniDA --metric H_score
python plotting/plot_curves.py --experiment PDA  --metric target_f1

# Bar versions of the same (closed-set has no n axis):
python plotting/plot_bars.py --experiment closed_set --metric target_f1
python plotting/plot_bars.py --experiment UniDA      --metric H_score

# Curriculum-order chip figure (hardest → easiest per pair):
python plotting/plot_curriculum_chips.py

# Per-method loss-curve diagnostics:
python plotting/plot_loss_curves.py --preset ppot_collapse
python plotting/plot_loss_curves.py --preset raincoat_phases

# Train-duration breakdown + t-SNE feature visualization:
python plotting/plot_train_duration.py
python plotting/make_tsne.py --source_dataset RealWorld --target_dataset Pamap2 \
                             --da_method UniJDOT --scenario UniDA --n_unknown 4

# LaTeX result tables for the thesis appendix:
python plotting/make_latex_tables.py        # writes tables/*.tex + extended_results.tex
```

## 5. Repository layout

```
algorithms/                 — every DA algorithm (one file)
configs/                    — per-dataset / per-hparam configs + sweep spaces
dataloader/                 — windowed-data loaders
models/                     — CNN / FNO backbones, RAINCOAT + TSFA blocks
preprocessing/              — raw HAR → windowed .pkl pipeline
trainers/                   — Trainer (single run) + sweep Trainer
analysis/                   — runs.csv + training_logs.csv + EDA outputs
plotting/                   — every plot/export/aggregation script
feature_distance_4known_fno_mean/      — OSDA ranking JSON
feature_distance_4known_fno_mean_pda/  — PDA ranking JSON
figures/                    — generated plots
tables/                     — generated LaTeX tables
bashes_and_logs/            — all sweep wrappers + their stdout logs
main.py                     — single-config trainer
main_sweep.py               — Optuna hyper-parameter sweep
extract_best_hparams.py     — sweep → best_hparams.json
run_curriculum.py           — n-private-class curriculum runner
best_hparams.json           — sweep artifact (not auto-loaded; copy into configs/hparams.py)
```
