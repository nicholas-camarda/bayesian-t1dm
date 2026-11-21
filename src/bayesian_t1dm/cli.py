"""The ``bayesian-t1dm`` command: download the pump data, build the dataset, and write the readout.

Download: ``collect`` (one window of at most 30 days), ``backfill`` (a date range in windows) and ``fetch-profile-edits``
(the pump's profile edits for archived windows). Cleaning: ``normalize-raw`` (rebuild normalized windows from the
archived responses) and ``build-dataset`` (merge, deduplicate and validate Tandem and Apple Health data). Result:
``readout``. ``refresh`` runs collect, build-dataset and readout in that order and stops at the first failure.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta

from .acquisition import (
    ExportWindow,
    StepLogger,
    TConnectSyncSourceClient,
    backfill_profile_edits,
    backfill_tandem_exports,
    collect_tandem_exports,
    load_local_env_file,
    load_tandem_credentials,
    normalize_tconnectsync_archive,
)
from .dataset import build_merged_dataset, merged_dir
from .observability import LoggingSession, setup_run_logging
from .paths import ProjectPaths

COLLECT_MAX_DAYS = 30


def build_parser() -> argparse.ArgumentParser:
    logging_parent = argparse.ArgumentParser(add_help=False)
    logging_parent.add_argument("--log-level", choices=["ERROR", "WARNING", "INFO", "DEBUG"], default="INFO")
    logging_parent.add_argument("--quiet", action=argparse.BooleanOptionalAction, default=False)
    logging_parent.add_argument("--unsafe-debug-logging", action=argparse.BooleanOptionalAction, default=False)

    parser = argparse.ArgumentParser(prog="bayesian-t1dm", parents=[logging_parent], description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=".", help="Project root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_command(name: str, help_text: str) -> argparse.ArgumentParser:
        return subparsers.add_parser(name, help=help_text, parents=[logging_parent])

    collect = add_command("collect", "Fetch one Tandem Source window (at most 30 days) through tconnectsync")
    collect.add_argument("--start-date", default=None, help="Window start (YYYY-MM-DD); default 29 days before the end")
    collect.add_argument("--end-date", default=None, help="Window end (YYYY-MM-DD); default today")
    collect.add_argument("--manifest", default=None)
    collect.add_argument("--report", default=None)
    collect.add_argument("--env-file", default=None, help="Optional .env file with Tandem credentials")
    collect.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)

    backfill = add_command("backfill", "Fetch Tandem Source windows over a date range through tconnectsync")
    backfill.add_argument("--start-date", required=True, help="Earliest requested date (YYYY-MM-DD)")
    backfill.add_argument("--end-date", required=True, help="Latest requested date (YYYY-MM-DD)")
    backfill.add_argument("--window-days", type=int, default=COLLECT_MAX_DAYS)
    backfill.add_argument("--direction", choices=["backward", "forward"], default="backward")
    backfill.add_argument("--manifest", default=None)
    backfill.add_argument("--report", default=None)
    backfill.add_argument("--env-file", default=None, help="Optional .env file with Tandem credentials")
    backfill.add_argument("--strict", action=argparse.BooleanOptionalAction, default=False)
    backfill.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)

    edits = add_command("fetch-profile-edits", "Fetch pump profile segment edits (event 68) for archived windows that lack them")
    edits.add_argument("--env-file", default=None, help="Optional .env file with Tandem credentials")

    normalize_raw = add_command("normalize-raw", "Rebuild normalized tconnectsync windows from the archived raw responses")
    normalize_raw.add_argument("--raw", default=None)
    normalize_raw.add_argument("--window-id", default=None)
    normalize_raw.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    normalize_raw.add_argument("--report", default=None)
    normalize_raw.add_argument("--manifest", default=None)

    build_dataset = add_command("build-dataset", "Merge, deduplicate and validate all raw Tandem and Apple Health data (only when a raw file changed)")
    build_dataset.add_argument("--raw", default=None)
    build_dataset.add_argument("--force", action=argparse.BooleanOptionalAction, default=False, help="Rebuild even if no raw file changed")

    readout = add_command("readout", "Write the personal readout page and its diagnostics workbook from the merged dataset")
    readout.add_argument("--raw", default=None)
    readout.add_argument("--output", default=None, help="Output folder (default: runtime/output/readout)")

    refresh = add_command("refresh", "Collect the last 30 days, rebuild the dataset and write the readout")
    refresh.add_argument("--collect", action=argparse.BooleanOptionalAction, default=True, help="Fetch the latest 30 days from Tandem Source first")
    refresh.add_argument("--env-file", default=None, help="Optional .env file with Tandem credentials")
    return parser


def _collect_or_backfill(args: argparse.Namespace, parser: argparse.ArgumentParser, paths: ProjectPaths, session: LoggingSession) -> int:
    with session.stage("acquisition.credentials", command=args.command):
        credentials = load_tandem_credentials(args.root, args.env_file)
    manifest_path = args.manifest or str(paths.cloud_raw / "tandem_export_manifest.csv")
    report_path = args.report or str(paths.output_source / "tandem_acquisition_summary.md")
    step_log = StepLogger(session=session)
    with session.stage("acquisition.run", manifest_path=manifest_path, report_path=report_path):
        with TConnectSyncSourceClient(paths, region=credentials.region, timezone=credentials.timezone, pump_serial=credentials.pump_serial) as client:
            if args.command == "collect":
                end_date = date.fromisoformat(args.end_date) if args.end_date else date.today()
                start_date = date.fromisoformat(args.start_date) if args.start_date else end_date - timedelta(days=COLLECT_MAX_DAYS - 1)
                if (end_date - start_date).days + 1 > COLLECT_MAX_DAYS:
                    parser.error(f"collect accepts at most a {COLLECT_MAX_DAYS}-day window; use backfill for larger ranges")
                collect_tandem_exports(client, [ExportWindow(start_date=start_date, end_date=end_date)], paths, credentials, manifest_path=manifest_path,
                                       report_path=report_path, resume=True, strict=args.strict, step_log=step_log)
            else:
                backfill_tandem_exports(client, start_date=args.start_date, end_date=args.end_date, workspace=paths, credentials=credentials, window_days=args.window_days,
                                        direction=args.direction, manifest_path=manifest_path, report_path=report_path, resume=args.resume, strict=args.strict, step_log=step_log)
    return 0


def _dispatch_command(*, args: argparse.Namespace, parser: argparse.ArgumentParser, paths: ProjectPaths, session: LoggingSession) -> int:
    if args.command in {"collect", "backfill"}:
        return _collect_or_backfill(args, parser, paths, session)

    if args.command == "fetch-profile-edits":
        credentials = load_tandem_credentials(args.root, args.env_file)
        with session.stage("acquisition.profile_edits"):
            with TConnectSyncSourceClient(paths, region=credentials.region, timezone=credentials.timezone, pump_serial=credentials.pump_serial) as client:
                written = backfill_profile_edits(client, credentials, paths)
        print(f"Wrote {len(written)} profile edit files; run build-dataset next.")
        return 0

    if args.command == "normalize-raw":
        raw = args.raw or str(paths.cloud_raw / "tconnectsync")
        report = args.report or str(paths.output_source / "normalize_raw_summary.md")
        manifest = args.manifest or str(paths.cloud_raw / "tandem_export_manifest.csv")
        with session.stage("normalize_raw.rebuild", raw_root=raw, report_path=report):
            normalize_tconnectsync_archive(paths, raw_root=raw, window_id=args.window_id, force=args.force, report_path=report, manifest_path=manifest)
        return 0

    if args.command == "build-dataset":
        raw = args.raw or str(paths.cloud_raw)
        with session.stage("dataset.build", raw=raw, force=args.force):
            build_merged_dataset(paths, raw_root=raw, force=args.force)
        print(f"Validation report: {merged_dir(raw) / 'validation_report.md'}")
        return 0

    if args.command == "readout":
        from .readout_report import run as run_readout

        with session.stage("readout.run"):
            return run_readout(args.raw, args.output, root=paths.root)

    if args.command == "refresh":
        base = ["--root", str(args.root)]
        steps = ([["collect", *(["--env-file", args.env_file] if args.env_file else [])]] if args.collect else []) + [["build-dataset"], ["readout"]]
        for argv in steps:
            code = main([*base, *argv])
            if code != 0:
                print(f"Refresh stopped: {argv[0]} exited with {code}.")
                return code
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    paths = ProjectPaths.from_root(args.root).ensure()
    # Settings such as TIMEZONE_NAME live in the repository's .env; the shell environment wins.
    for key, value in load_local_env_file(paths.root / ".env").items():
        os.environ.setdefault(key, value)
    session = setup_run_logging(paths, command=args.command, argv=list(argv) if argv is not None else sys.argv[1:], log_level=args.log_level, quiet=args.quiet,
                                unsafe_debug_logging=args.unsafe_debug_logging)
    session.start()
    session.log_event("command.start", message=f"{args.command} started", status="started", root=str(paths.root))
    if session.startup_warning is not None:
        session.log_event("command.warning", level="WARNING", message=session.startup_warning, status="degraded")
    try:
        exit_code = _dispatch_command(args=args, parser=parser, paths=paths, session=session)
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else 1
        if not session.error_logged:
            session.log_event("command.error", level="ERROR", message=str(exc), stage=session.current_stage, status="failed", error_type="SystemExit", exit_code=exit_code)
        session.finalize(exit_code=exit_code, status="failed")
        raise
    except Exception as exc:
        if not session.error_logged:
            session.log_event("command.error", level="ERROR", message=str(exc), stage=session.current_stage, status="failed", error_type=type(exc).__name__)
        session.finalize(exit_code=1, status="failed")
        raise
    session.finalize(exit_code=exit_code, status="success" if exit_code == 0 else "failed")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
