# Bayesian T1DM

A personal glucose readout built from a Tandem insulin pump and Apple Health. It answers three questions on one phone-sized page: how did the last day and week go, what has gone with my glucose (sleep, recovery, exercise, late meals, infusion-site age), and are my basal and I:C settings performing? For example: "days after a workout had about 2 points more of the day in range", or "overnight basal: Control-IQ delivered 40% above the programmed rate, so the basal looks low".

It is decision support for a conversation with an endocrinologist, not a medical device; nothing here changes a pump setting.

## Outputs

- `readout.html`: rings for the latest full day, this week against your usual range, the driver effects with 90% intervals, the settings check by pump segment, recent setting changes, and the last seven days as glucose traces with sleep, workouts, carbs and site changes. The page is copied to the Project Vault (`readout/readout.html`) so it opens from the OneDrive app.
- `diagnostics.xlsx` beside it: the specification, the day flow, effects, sampler health, the settings evidence and every left-out day with its reason.

The effects are associations within one person's record; they say what went with what, not what caused it. Methods are in [`docs/methods.md`](docs/methods.md); each step's rules and failure modes are in [`docs/technical.md`](docs/technical.md).

## Requirements

- macOS or Linux with Conda (Python 3.11 environment from `environment.yml`)
- A Tandem Source account (pump data are pulled with [`tconnectsync`](https://github.com/jwoglom/tconnectsync) 3.x)
- Apple Health exports from the Health Auto Export app (JSON), for sleep, heart rate, HRV and workouts
- The Bayesian model runs on PyTensor's Numba backend, set by `environment.yml` (`PYTENSOR_FLAGS=mode=NUMBA,cxx=`); outside conda, export that variable yourself

## Install

```bash
conda env create -f environment.yml
```

```bash
conda activate bt1dm
```

Create a `.env` file in the repository root (never committed) from the template and fill in your Tandem credentials and time zone:

```bash
cp .env.example .env
```

| Variable | Holds | Default |
| --- | --- | --- |
| `BAYESIAN_T1DM_CLOUD_ROOT` | Raw data, the merged dataset and the readout page | a `Project Vault` folder under OneDrive |
| `BAYESIAN_T1DM_RUNTIME_ROOT` | Generated work (`output/`, `logs/`) | `~/Workspaces/bayesian-t1dm/runtime` |

## First run

```bash
bayesian-t1dm backfill --start-date 2024-01-01 --end-date 2026-11-17
```

```bash
bayesian-t1dm build-dataset
```

```bash
bayesian-t1dm readout
```

`backfill` saves Tandem Source responses untouched under `data/raw/tconnectsync/<window>/raw/` in the cloud root. Drop Health Auto Export JSON files in `data/raw/health_auto_export/`. `build-dataset` merges everything into one validated dataset and only does work when a raw file changed; the readout refuses to run on a dataset older than its raw files.

## Usage

| Step | Command |
| --- | --- |
| Fetch the last 30 days from Tandem Source | `collect` |
| Fetch a date range | `backfill --start-date ... --end-date ...` |
| Fetch the pump's profile edits for windows archived without them | `fetch-profile-edits` |
| Rebuild normalized tables from archived responses | `normalize-raw` |
| Merge and validate all raw data | `build-dataset` |
| Write the readout page and workbook | `readout` |
| Collect, build and write the readout in one go | `refresh` (add `--no-collect` to skip the download) |

Optional inputs under `data/raw/` in the cloud root:

| Path | What goes there |
| --- | --- |
| `health_auto_export/` | Apple Health exports: `HealthAutoExport-*.json` files |
| `carbs.csv` | Hand-kept carb log, `timestamp,carb_grams` in local pump time |
| `context_events.csv` | Days to leave out, `start_date,end_date,reason` (for example illness or a steroid injection) |

## Limits

One person's retrospective record under a closed-loop pump supports associations, not causes. Tandem Source is undocumented and has changed without notice; when collection fails, check for a newer `tconnectsync`. Carbohydrate and I:C cannot be separated from glucose alone, so I:C evidence needs logged carbs.

## Development

```bash
python -m pytest -q
```

```bash
python -m ruff check src tests
```

Both run in GitHub Actions on pushes to `main` and on pull requests. Tests use synthetic data only.

## License and contact

Personal research code; no license is granted. Contact the repository owner through GitHub.
