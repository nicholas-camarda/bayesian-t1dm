# Repo Guidance

## Project Shape

- Active code lives in [`bayesian-t1dm.R`](./bayesian-t1dm.R) and [`helper_functions.R`](./helper_functions.R).
- Treat [`old/`](./old) as archive or scratch space unless a task explicitly asks to revive it.
- The current scripts are not packaged; there is no `DESCRIPTION`, `renv.lock`, or formal test harness yet.

## Working Rules

- Prefer `Rscript` for quick validation and parsing checks.
- Before changing analysis logic, inspect the actual data flow and confirm which objects are defined locally versus inherited from archived scripts.
- Be skeptical of script success alone; check parseability, runtime dependencies, and whether the analysis is standalone.

## Current Risks To Watch

- [`bayesian-t1dm.R`](./bayesian-t1dm.R) currently has a parse error in the `complete_dataset` pipeline.
- The top-level script depends on `temp_df`, but that object is only constructed in archived code under [`old/scripts/`](./old/scripts).
- `helper_functions.R` relies on tidyverse-style verbs and `qq()` when plotting, so its implicit package dependencies should be made explicit if the file is used standalone.

## Suggested Validation

- Run `Rscript -e "parse(file = 'bayesian-t1dm.R')"` after any edit to the main script.
- Run `Rscript -e "parse(file = 'helper_functions.R')"` after helper edits.
- If the archived analysis is being touched, validate the corresponding script separately and do not assume top-level objects exist.

## Repo Hygiene

- Keep generated artifacts out of version control.
- Do not edit archive files under `old/` unless the task specifically requires it.
- Prefer small, reviewable changes that preserve the current exploratory workflow until a deliberate refactor is planned.
