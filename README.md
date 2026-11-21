# Bayesian T1DM

A personal research workbench that turns Tandem insulin-pump data and Apple Health data into two things: a forecast of 14-day time in range (70-180 mg/dL), and a test of what the record can show about changing pump settings. For example: does a Bayesian model of every hour of the record predict the next fortnight's time in range better than the last 28 days do, and do its predicted effects of past setting changes match what followed them?

It is research and decision support, not a medical device; nothing here changes a pump setting.

## Results

`refresh` writes a front page (`index.html` in the current run folder) that gives the verdict on each question, the evidence behind it and the data-quality status. Read that page and the generated reports for results; this repository states no run numbers.

- Predictive and associational claims only: the record is retrospective and from one person.
- The hour-day model beats the prespecified last-28-day baseline but ties a plain 365-day mean, so the win comes from long-run level, not from the settings.
- Effects of settings are not validated: predicted effects of past changes do not track the observed ones, with few isolated changes to test on.
- Ranked candidate changes are not produced, because that step is gated on a validated settings effect.

## Requirements

- macOS or Linux with Conda (Python 3.11 environment, created from `environment.yml`)
- A Tandem Source account for pump data (pulled with [`tconnectsync`](https://github.com/jwoglom/tconnectsync) 3.x)
- Optional: Apple Health exports from the Health Auto Export app (JSON)
- The Bayesian models run on PyTensor's Numba backend, set by `environment.yml` (`PYTENSOR_FLAGS=mode=NUMBA,cxx=`); no C compiler is needed. Outside conda, export that variable yourself.

## Install

```bash
conda env create -f environment.yml
```

```bash
conda activate bt1dm
```

Create a `.env` file in the repository root (never committed) from the template, and fill in your Tandem credentials and time zone:

```bash
cp .env.example .env
```

Two folders are used; set them in `.env` or the environment if the defaults do not suit you:

| Variable | Holds | Default |
| --- | --- | --- |
| `BAYESIAN_T1DM_CLOUD_ROOT` | Raw data, merged dataset, change record, published runs | a `Project Vault` folder under OneDrive |
| `BAYESIAN_T1DM_RUNTIME_ROOT` | Generated reports and run folders (`output/`, `runs/`, `current/`) | `~/Workspaces/bayesian-t1dm/runtime` |

## First run

```bash
bayesian-t1dm --help
```

```bash
bayesian-t1dm backfill --start-date 2024-01-01 --end-date 2026-11-17
```

```bash
bayesian-t1dm build-dataset
```

```bash
bayesian-t1dm status
```

`backfill` saves Tandem Source responses untouched under `data/raw/tconnectsync/<window>/raw/` in the cloud root. `build-dataset` merges all Tandem and Apple Health data into one deduplicated, validated dataset; it does work only when a raw file changed, and every analysis refuses to run on a dataset older than its raw files.

## Usage

Commands are `bayesian-t1dm <command>`; the research analyses run as modules (`python -m bayesian_t1dm.<name>`). Each writes under `<runtime root>/output/<folder>/`.

| Question | Command | Output folder |
| --- | --- | --- |
| Run everything and publish a run to the Vault | `refresh` (add `--no-collect --no-publish` to skip the Tandem fetch and the publish) | `runs/<id>/`, copied to `current/` |
| Rewrite the front page of a run without a refresh | `front-page` | `current/index.html` |
| Is the record fit to be modelled? | `python -m bayesian_t1dm.data_audit` | `data_audit/` |
| Does the model forecast 14-day time in range better than the last 28 days? | `python -m bayesian_t1dm.forecast_harness` | `forecast_harness/` |
| How did the last day and week go, what has gone with my glucose, and are my settings performing? | `python -m bayesian_t1dm.readout_report` (page also copied to the Vault `readout/` folder) | `readout/` |
| Do the model's predicted effects of past changes match what happened? | `python -m bayesian_t1dm.replay_predictor` | `replay_predictor/` |
| Do outcomes after past changes differ from placebo dates? (exploratory) | `python -m bayesian_t1dm.replay` | `replay/` |
| Which hour-day covariates are associated with time in range? | `python -m bayesian_t1dm.outcomes_model` | `outcomes_model/` |
| Where do time in range, lows and highs stand? | `report-outcomes` | `outcomes/` |
| Which breakfast I:C change does the evidence support? | `propose-breakfast-icr` | `proposals/` |
| Is the Tandem data complete? | `status`, `ingest` | `current_status.html`, `source/` |

The harness and replay-predictor fit many Bayesian models and take tens of minutes; the others take seconds to minutes. Methods, goals and how to read each result are in [`docs/methods.md`](docs/methods.md); implementation detail is in [`docs/technical.md`](docs/technical.md); scientific status is in [`docs/TECHNICAL_LAB_NOTEBOOK.md`](docs/TECHNICAL_LAB_NOTEBOOK.md).

Optional inputs under `data/raw/` in the cloud root:

| Path | What goes there |
| --- | --- |
| `health_auto_export/` | Apple Health exports: drop `HealthAutoExport-*.json` files anywhere here |
| `carbs.csv` | Hand-kept carb log, `timestamp,carb_grams` in local pump time |
| `context_events.csv` | Days to leave out of meal and basal analyses, `start_date,end_date,reason` (for example a steroid injection) |

## Limits

Retrospective personal data support predictive and associational claims, not causal ones; results are for a conversation with a clinician. Tandem Source is undocumented and has changed without notice; when collection fails, check for a newer `tconnectsync`. Carbs and I:C cannot be separated from glucose alone, so I:C evidence needs logged carbs.

## Development

```bash
python -m pytest -q
```

```bash
python -m ruff check src tests
```

Both run in GitHub Actions on pushes to `main` and on pull requests. Tests use synthetic data only; see [`docs/technical.md`](docs/technical.md#validation) for the validation design.

## License and contact

Personal research code; no license is granted. Contact the repository owner through GitHub.
