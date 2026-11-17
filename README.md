# Bayesian T1DM

A personal research workbench that turns Tandem insulin-pump data and Apple Health data into evidence about pump settings. For example: given every breakfast logged on the pump, does glucose respond as if the morning insulin-to-carb (I:C) ratio were right?

It is research and decision support, not a medical device; nothing here changes a pump setting.

## Questions it answers

| Question | Command | Answer page |
| --- | --- | --- |
| Is my morning I:C ratio right? | `research-latent-meal-icr --research-scope full` | `output/latent_meal/latent_meal_review.html` |
| Is there enough clean overnight data to judge basal? | `review-therapy-evidence` | `output/therapy/therapy_review.html` |
| Can glucose be forecast 30 minutes ahead better than "it stays the same"? | `run --skip-recommendations` | `output/forecast/forecast_review.html` |
| Is the Tandem data complete? | `ingest` | `output/source/coverage_review.html` |
| Where does everything stand? | `status` | `output/current_status.html` |

Every page opens with the question, a one-sentence answer, how much to trust it and what to do next; figures, tables and the files behind it fold out below. Methods are in [`docs/technical.md`](docs/technical.md); current scientific status is in [`docs/TECHNICAL_LAB_NOTEBOOK.md`](docs/TECHNICAL_LAB_NOTEBOOK.md).

## Requirements

- macOS or Linux with Conda
- A Tandem Source account (pump data is pulled with [`tconnectsync`](https://github.com/jwoglom/tconnectsync) 3.x)
- Optional: Apple Health exports from the Health Auto Export app (JSON)

## Install

```bash
conda env create -f environment.yml
```

```bash
conda activate bt1dm
```

Put Tandem credentials in a `.env` file in the repository root (never committed):

```bash
TANDEM_SOURCE_EMAIL=you@example.com
TANDEM_SOURCE_PASSWORD=...
TIMEZONE_NAME=America/New_York
```

## First run

```bash
bayesian-t1dm backfill --start-date 2024-01-01 --end-date 2026-11-17
```

```bash
bayesian-t1dm build-dataset
```

```bash
bayesian-t1dm status
```

`backfill` saves Tandem Source responses untouched under `data/raw/tconnectsync/<window>/raw/` and cleaned tables next to them. `build-dataset` merges all Tandem and Apple Health data into one deduplicated, validated dataset; it does work only when a raw file changed, and every analysis refuses to run on a dataset older than its raw files.

## Data

Everything lives in the project folder set by `BAYESIAN_T1DM_CLOUD_ROOT` (the Project Vault by default); generated reports go to `~/Workspaces/bayesian-t1dm/runtime/output/`.

| Path under the project folder | What goes there |
| --- | --- |
| `data/raw/tconnectsync/` | Tandem downloads, one folder per 30-day window (written by `collect` / `backfill`) |
| `data/raw/health_auto_export/` | Apple Health exports: drop `HealthAutoExport-*.json` files anywhere here |
| `data/raw/carbs.csv` | Optional hand-kept carb log, `timestamp,carb_grams` in local pump time |
| `data/raw/context_events.csv` | Days to leave out of meal and basal analyses, `start_date,end_date,reason` (for example a steroid injection) |
| `data/merged/` | The merged dataset, `validation_report.md` and input fingerprints (written by `build-dataset`) |

Carbs entered in the pump's bolus calculator are read from the pump events, together with the I:C ratio and correction factor the calculator used, so each meal is judged against the setting in force that day. The latent-meal analysis reads your programmed profile from the newest Tandem window; `--therapy-settings file.json` overrides it with this format:

```json
{
  "icr_grams_per_unit": {"overnight": 12.0, "morning": 10.0, "afternoon": 11.0, "evening": 9.0},
  "correction_factor_mgdl_per_unit": {"overnight": 45.0, "morning": 40.0, "afternoon": 45.0, "evening": 40.0},
  "target_glucose_mgdl": 110.0,
  "insulin_action_minutes": 300,
  "carb_absorption_minutes_prior": 180,
  "max_plausible_meal_carbs": 180
}
```

## Limits

Tandem Source is undocumented and has changed without notice; when collection fails, check for a newer `tconnectsync`. Carbs and I:C cannot be separated from glucose alone, so I:C evidence needs logged carbs. Retrospective personal data support associations, not causal claims; results are for a conversation with a clinician.

## Development

```bash
pytest -q
```

```bash
ruff check src tests
```

Both run in GitHub Actions on pushes to `main` and on pull requests. Tests use synthetic data only. `bayesian-t1dm validate-latent-meal` checks the carb estimator against simulated meals with known carbohydrate.

## License and contact

Personal research code; no license is granted. Contact the repository owner through GitHub.
