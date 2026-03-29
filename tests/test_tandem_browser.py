from __future__ import annotations

from bayesian_t1dm.tandem_browser import (
    LocatorSpec,
    TandemPageMap,
    discover_export_confirm_from_controls,
    discover_login_controls_from_controls,
    discover_tandem_page_map_from_controls,
    discover_timeline_controls_from_controls,
)


def _snapshot() -> dict[str, object]:
    return {
        "role": "document",
        "children": [
            {"role": "textbox", "name": "Email address"},
            {"role": "textbox", "name": "Password"},
            {"role": "button", "name": "Sign In"},
            {"role": "link", "name": "Daily Timeline"},
            {"role": "combobox", "name": "2 Weeks (Mar 16 - 29, 2026)"},
            {"role": "button", "name": "Select"},
            {"role": "button", "name": "Export CSV"},
        ],
    }


def _inventory() -> list[dict[str, object]]:
    return [
        {"tag": "input", "role": None, "id": "email", "name": "email", "type": "email", "aria_label": "Email address", "placeholder": None, "autocomplete": "username", "text": "", "title": None, "href": None, "data_testid": None, "visible": True},
        {"tag": "input", "role": None, "id": "password", "name": "password", "type": "password", "aria_label": "Password", "placeholder": None, "autocomplete": "current-password", "text": "", "title": None, "href": None, "data_testid": None, "visible": True},
        {"tag": "button", "role": None, "id": "submit", "name": None, "type": "submit", "aria_label": None, "placeholder": None, "autocomplete": None, "text": "Sign In", "title": None, "href": None, "data_testid": None, "visible": True},
        {"tag": "a", "role": None, "id": "timeline", "name": None, "type": None, "aria_label": None, "placeholder": None, "autocomplete": None, "text": "Daily Timeline", "title": None, "href": "/daily", "data_testid": None, "visible": True},
        {"tag": "div", "role": "combobox", "id": "range", "name": None, "type": None, "aria_label": None, "placeholder": None, "autocomplete": None, "text": "2 Weeks (Mar 16 - 29, 2026)", "title": None, "href": None, "data_testid": None, "visible": True},
        {"tag": "button", "role": None, "id": "select", "name": None, "type": "button", "aria_label": None, "placeholder": None, "autocomplete": None, "text": "Select", "title": None, "href": None, "data_testid": None, "visible": True},
        {"tag": "button", "role": None, "id": "export", "name": None, "type": "button", "aria_label": None, "placeholder": None, "autocomplete": None, "text": "Export CSV", "title": None, "href": None, "data_testid": None, "visible": True},
    ]


def _export_modal_snapshot() -> dict[str, object]:
    return {
        "role": "document",
        "children": [
            {
                "role": "dialog",
                "name": "Export to CSV",
                "children": [
                    {"role": "button", "name": "Cancel"},
                    {"role": "button", "name": "Export"},
                ],
            }
        ],
    }


def _export_modal_inventory() -> list[dict[str, object]]:
    return [
        {"tag": "div", "role": "dialog", "id": "export-modal", "name": None, "type": None, "aria_label": None, "placeholder": None, "autocomplete": None, "text": "Export to CSV Cancel Export", "title": None, "href": None, "data_testid": None, "visible": True},
        {"tag": "button", "role": None, "id": "cancel", "name": None, "type": "button", "aria_label": None, "placeholder": None, "autocomplete": None, "text": "Cancel", "title": None, "href": None, "data_testid": None, "visible": True},
        {"tag": "button", "role": None, "id": "confirm", "name": None, "type": "button", "aria_label": None, "placeholder": None, "autocomplete": None, "text": "Export", "title": None, "href": None, "data_testid": None, "visible": True},
    ]


def test_discover_login_and_timeline_controls_from_controls():
    login_email, login_password, login_submit = discover_login_controls_from_controls(
        accessibility_snapshot=_snapshot(),
        control_inventory=_inventory(),
    )
    daily_timeline_nav, start_date, end_date, export_csv = discover_timeline_controls_from_controls(
        accessibility_snapshot=_snapshot(),
        control_inventory=_inventory(),
    )

    assert login_email.kind == "css"
    assert login_email.selector == "#email"
    assert login_password.kind == "css"
    assert login_password.selector == "#password"
    assert login_submit.kind == "css"
    assert login_submit.selector == "#submit"
    assert daily_timeline_nav.kind == "role"
    assert daily_timeline_nav.role == "link"
    assert daily_timeline_nav.name == "Daily Timeline"
    assert start_date.kind == "role"
    assert start_date.role == "combobox"
    assert end_date.kind == "role"
    assert end_date.role == "button"
    assert end_date.name == "Select"
    assert export_csv.kind == "role"
    assert export_csv.role == "button"
    assert export_csv.name == "Export CSV"


def test_discover_export_confirm_prefers_modal_dialog():
    confirm = discover_export_confirm_from_controls(
        accessibility_snapshot=_export_modal_snapshot(),
        control_inventory=_export_modal_inventory(),
    )

    assert confirm.kind == "role"
    assert confirm.role == "button"
    assert confirm.name == "Export"


def test_locator_spec_frame_roundtrip():
    spec = LocatorSpec(kind="css", selector="#export", frame_name="daily", frame_url="https://source.example/daily")
    loaded = LocatorSpec.from_dict(spec.to_dict())
    assert loaded == spec


def test_discover_tandem_page_map_roundtrip(tmp_path):
    page_map = discover_tandem_page_map_from_controls(
        accessibility_snapshot=_snapshot(),
        control_inventory=_inventory(),
        login_url="https://source.tandemdiabetes.com/",
        daily_timeline_url=None,
    )

    saved = page_map.save(tmp_path / "tandem_page_map.json")
    loaded = TandemPageMap.load(saved)

    assert isinstance(loaded.login_email, LocatorSpec)
    assert loaded == page_map
    assert loaded.export_csv.role == "button"
    assert loaded.daily_timeline_nav.name == "Daily Timeline"


def test_legacy_page_map_uses_export_csv_as_launcher():
    legacy = TandemPageMap.from_dict(
        {
            "login_url": "https://source.tandemdiabetes.com/",
            "login_email": {"kind": "css", "selector": "#email"},
            "login_password": {"kind": "css", "selector": "#password"},
            "login_submit": {"kind": "css", "selector": "#submit"},
            "daily_timeline_nav": {"kind": "role", "role": "link", "name": "Daily Timeline"},
            "start_date": {"kind": "role", "role": "combobox", "name": "Custom"},
            "end_date": {"kind": "role", "role": "button", "name": "Select"},
            "export_csv": {"kind": "role", "role": "button", "name": "Export CSV"},
        }
    )

    assert legacy.export_csv_launcher is not None
    assert legacy.export_csv_launcher.name == "Export CSV"
    assert legacy.export_csv_confirm is None
