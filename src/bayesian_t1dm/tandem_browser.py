from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

DEFAULT_LOGIN_URL = "https://source.tandemdiabetes.com/"


@dataclass(frozen=True)
class LocatorSpec:
    kind: str
    selector: str | None = None
    role: str | None = None
    name: str | None = None
    exact: bool = True
    frame_name: str | None = None
    frame_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LocatorSpec":
        return cls(
            kind=str(data["kind"]),
            selector=data.get("selector"),
            role=data.get("role"),
            name=data.get("name"),
            exact=bool(data.get("exact", True)),
            frame_name=data.get("frame_name"),
            frame_url=data.get("frame_url"),
        )

    def _resolve_frame(self, page):
        if not self.frame_name and not self.frame_url:
            return page.main_frame
        candidates = list(page.frames)
        if self.frame_name:
            for frame in candidates:
                if frame.name == self.frame_name:
                    return frame
        if self.frame_url:
            for frame in candidates:
                if frame.url == self.frame_url or self.frame_url in frame.url:
                    return frame
        return page.main_frame

    def locate(self, page):
        frame = self._resolve_frame(page)
        if self.kind == "role":
            if not self.role:
                raise ValueError("role locator requires a role")
            return frame.get_by_role(self.role, name=self.name, exact=self.exact)
        if self.kind == "css":
            if not self.selector:
                raise ValueError("css locator requires a selector")
            return frame.locator(self.selector)
        if self.kind == "label":
            if not self.selector:
                raise ValueError("label locator requires a selector")
            return frame.get_by_label(self.selector, exact=self.exact)
        if self.kind == "placeholder":
            if not self.selector:
                raise ValueError("placeholder locator requires a selector")
            return frame.get_by_placeholder(self.selector, exact=self.exact)
        if self.kind == "text":
            if not self.selector:
                raise ValueError("text locator requires a selector")
            return frame.get_by_text(self.selector, exact=self.exact)
        raise ValueError(f"Unsupported locator kind: {self.kind}")

    def describe(self) -> str:
        if self.kind == "role":
            return f"role={self.role!r} name={self.name!r} frame={self.frame_name or self.frame_url or 'main'}"
        if self.kind in {"css", "label", "placeholder", "text"}:
            return f"{self.kind}={self.selector!r} frame={self.frame_name or self.frame_url or 'main'}"
        return f"{self.kind}:{self.selector or self.role or self.name or ''}"


@dataclass(frozen=True)
class TandemPageMap:
    login_url: str = DEFAULT_LOGIN_URL
    daily_timeline_url: str | None = None
    login_email: LocatorSpec | None = None
    login_password: LocatorSpec | None = None
    login_submit: LocatorSpec | None = None
    daily_timeline_nav: LocatorSpec | None = None
    start_date: LocatorSpec | None = None
    end_date: LocatorSpec | None = None
    export_csv: LocatorSpec | None = None
    generated_at: str | None = None
    source: str = "playwright-discovery"

    def validate(self) -> None:
        missing = [
            name
            for name, value in self.to_locator_dict().items()
            if value is None
        ]
        if missing:
            raise ValueError(f"Page map is incomplete; missing: {', '.join(missing)}")

    def is_complete(self) -> bool:
        return all(value is not None for value in self.to_locator_dict().values())

    def to_locator_dict(self) -> dict[str, LocatorSpec | None]:
        return {
            "login_email": self.login_email,
            "login_password": self.login_password,
            "login_submit": self.login_submit,
            "daily_timeline_nav": self.daily_timeline_nav,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "export_csv": self.export_csv,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "login_url": self.login_url,
            "daily_timeline_url": self.daily_timeline_url,
            "login_email": self.login_email.to_dict() if self.login_email else None,
            "login_password": self.login_password.to_dict() if self.login_password else None,
            "login_submit": self.login_submit.to_dict() if self.login_submit else None,
            "daily_timeline_nav": self.daily_timeline_nav.to_dict() if self.daily_timeline_nav else None,
            "start_date": self.start_date.to_dict() if self.start_date else None,
            "end_date": self.end_date.to_dict() if self.end_date else None,
            "export_csv": self.export_csv.to_dict() if self.export_csv else None,
            "generated_at": self.generated_at,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TandemPageMap":
        def load_spec(key: str) -> LocatorSpec | None:
            value = data.get(key)
            if value is None:
                return None
            return LocatorSpec.from_dict(value)

        return cls(
            login_url=str(data.get("login_url", DEFAULT_LOGIN_URL)),
            daily_timeline_url=data.get("daily_timeline_url"),
            login_email=load_spec("login_email"),
            login_password=load_spec("login_password"),
            login_submit=load_spec("login_submit"),
            daily_timeline_nav=load_spec("daily_timeline_nav"),
            start_date=load_spec("start_date"),
            end_date=load_spec("end_date"),
            export_csv=load_spec("export_csv"),
            generated_at=data.get("generated_at"),
            source=str(data.get("source", "playwright-discovery")),
        )

    def save(self, path: str | Path, *, validate: bool = True) -> Path:
        if validate:
            self.validate()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "TandemPageMap":
        source = Path(path)
        data = json.loads(source.read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass(frozen=True)
class PageDiagnostics:
    html_path: Path
    accessibility_path: Path
    inventory_path: Path
    url_path: Path
    frames_path: Path
    screenshot_path: Path


def _normalize_text(value: str | None) -> str:
    return " ".join((value or "").split()).strip()


def _capture_frame_control_inventory(frame, *, frame_index: int, is_main: bool) -> list[dict[str, Any]]:
    js = """
() => {
  const controls = Array.from(document.querySelectorAll('input, button, a, textarea, select, [role]'));
  return controls.map((el, index) => {
    const rect = el.getBoundingClientRect();
    return {
      index,
      tag: el.tagName.toLowerCase(),
      role: el.getAttribute('role'),
      id: el.id || null,
      name: el.getAttribute('name') || null,
      type: el.getAttribute('type') || null,
      aria_label: el.getAttribute('aria-label') || null,
      placeholder: el.getAttribute('placeholder') || null,
      autocomplete: el.getAttribute('autocomplete') || null,
      text: (el.innerText || el.textContent || '').replace(/\\s+/g, ' ').trim(),
      title: el.getAttribute('title') || null,
      href: el.getAttribute('href') || null,
      data_testid: el.getAttribute('data-testid') || null,
      visible: !!(rect.width && rect.height),
    };
  });
}
    """
    controls = list(frame.evaluate(js))
    for control in controls:
        control["frame_index"] = frame_index
        control["frame_name"] = getattr(frame, "name", None)
        control["frame_url"] = getattr(frame, "url", None)
        control["frame_is_main"] = is_main
    return controls


def capture_control_inventory(page) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    frames = list(page.frames)
    for frame_index, frame in enumerate(frames):
        try:
            inventory.extend(
                _capture_frame_control_inventory(
                    frame,
                    frame_index=frame_index,
                    is_main=(frame == page.main_frame),
                ),
            )
        except Exception:
            continue
    return inventory


def _coerce_ax_value(value: Any) -> str | None:
    if isinstance(value, dict):
        return _normalize_text(value.get("value") if isinstance(value.get("value"), str) else None)
    if isinstance(value, str):
        return _normalize_text(value)
    return None


def _normalize_ax_node(node: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {
        "role": _coerce_ax_value(node.get("role")) or "",
        "name": _coerce_ax_value(node.get("name")) or "",
        "value": _coerce_ax_value(node.get("value")),
        "ignored": bool(node.get("ignored", False)),
        "children": [],
    }
    return normalized


def _capture_accessibility_snapshot(page) -> dict[str, Any] | None:
    try:
        session = page.context.new_cdp_session(page)
        response = session.send("Accessibility.getFullAXTree")
        nodes = response.get("nodes") or []
        by_id = {node.get("nodeId"): node for node in nodes if node.get("nodeId") is not None}
        child_ids = {child_id for node in nodes for child_id in node.get("childIds") or []}
        root_nodes = [node for node in nodes if node.get("nodeId") not in child_ids]

        def build(node: dict[str, Any]) -> dict[str, Any]:
            normalized = _normalize_ax_node(node)
            normalized["children"] = [
                build(by_id[child_id])
                for child_id in node.get("childIds") or []
                if child_id in by_id
            ]
            return normalized

        return {
            "source": "cdp",
            "nodes": [build(node) for node in root_nodes],
        }
    except Exception:
        try:
            return {
                "source": "aria_snapshot",
                "text": page.locator("body").aria_snapshot(timeout=5_000),
            }
        except Exception:
            return None


def capture_accessibility_snapshot(page) -> dict[str, Any] | None:
    return _capture_accessibility_snapshot(page)


def capture_page_diagnostics(page, output_dir: str | Path, stem: str) -> PageDiagnostics:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    html_path = output / f"{stem}.html"
    accessibility_path = output / f"{stem}.a11y.json"
    inventory_path = output / f"{stem}.controls.json"
    url_path = output / f"{stem}.url.txt"
    frames_path = output / f"{stem}.frames.json"
    screenshot_path = output / f"{stem}.png"

    html_path.write_text(page.content(), encoding="utf-8")
    a11y = _capture_accessibility_snapshot(page)
    accessibility_path.write_text(json.dumps(a11y, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    inventory = capture_control_inventory(page)
    inventory_path.write_text(json.dumps(inventory, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    url_path.write_text(page.url + "\n", encoding="utf-8")
    frames = [
        {
            "name": frame.name,
            "url": frame.url,
        }
        for frame in page.frames
    ]
    frames_path.write_text(json.dumps(frames, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    page.screenshot(path=str(screenshot_path), full_page=True)
    return PageDiagnostics(
        html_path=html_path,
        accessibility_path=accessibility_path,
        inventory_path=inventory_path,
        url_path=url_path,
        frames_path=frames_path,
        screenshot_path=screenshot_path,
    )


def _walk_a11y(node: dict[str, Any] | None) -> Iterable[dict[str, Any]]:
    if not node:
        return
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        children = current.get("children") or []
        stack.extend(reversed(children))


def _find_node(snapshot: dict[str, Any] | None, *, roles: Sequence[str], keywords: Sequence[str]) -> dict[str, Any] | None:
    lowered = [keyword.lower() for keyword in keywords]
    for node in _walk_a11y(snapshot):
        role = str(node.get("role") or "").lower()
        name = _normalize_text(node.get("name") if isinstance(node.get("name"), str) else None)
        if role in {r.lower() for r in roles} and name:
            haystack = name.lower()
            if any(keyword in haystack for keyword in lowered):
                return node
    return None


def _find_inventory_entry(inventory: Sequence[dict[str, Any]], *, keywords: Sequence[str], roles: Sequence[str] | None = None) -> dict[str, Any] | None:
    lowered = [keyword.lower() for keyword in keywords]
    role_set = {r.lower() for r in roles} if roles else None
    for entry in inventory:
        tag = str(entry.get("tag") or "").lower()
        entry_role = str(entry.get("role") or "").lower()
        entry_type = str(entry.get("type") or "").lower()
        visible = bool(entry.get("visible", True))
        if not visible:
            continue
        derived_role = entry_role
        if not derived_role:
            if tag in {"a"}:
                derived_role = "link"
            elif tag in {"button"}:
                derived_role = "button"
            elif tag in {"input", "textarea"}:
                derived_role = "textbox"
            elif tag in {"select"}:
                derived_role = "combobox"
        if role_set and derived_role not in role_set:
            continue
        texts = [
            str(entry.get("aria_label") or ""),
            str(entry.get("placeholder") or ""),
            str(entry.get("text") or ""),
            str(entry.get("name") or ""),
            str(entry.get("title") or ""),
            str(entry.get("id") or ""),
            str(entry.get("autocomplete") or ""),
            entry_type,
        ]
        haystack = " ".join(texts).lower()
        if any(keyword in haystack for keyword in lowered):
            return dict(entry)
    return None


def _spec_from_inventory_entry(entry: dict[str, Any]) -> LocatorSpec:
    role = str(entry.get("role") or "").strip()
    text = _normalize_text(
        entry.get("aria_label")
        or entry.get("placeholder")
        or entry.get("text")
        or entry.get("title")
        or entry.get("name")
        or entry.get("id")
    )
    if role and text:
        return LocatorSpec(kind="role", role=role, name=text, exact=True)
    if entry.get("id"):
        return LocatorSpec(kind="css", selector=f"#{entry['id']}")
    if entry.get("name"):
        tag = str(entry.get("tag") or "input").lower()
        return LocatorSpec(kind="css", selector=f'{tag}[name="{entry["name"]}"]')
    if entry.get("placeholder"):
        return LocatorSpec(kind="placeholder", selector=text, exact=True)
    if entry.get("aria_label"):
        return LocatorSpec(kind="label", selector=text, exact=True)
    raise ValueError(f"Could not derive a stable selector from control entry: {entry}")


def _discover_spec(
    *,
    snapshot: dict[str, Any] | None,
    inventory: Sequence[dict[str, Any]],
    roles: Sequence[str],
    keywords: Sequence[str],
) -> LocatorSpec:
    node = _find_node(snapshot, roles=roles, keywords=keywords)
    if node and _normalize_text(node.get("name") if isinstance(node.get("name"), str) else None):
        return LocatorSpec(kind="role", role=str(node["role"]), name=_normalize_text(node["name"]), exact=True)
    entry = _find_inventory_entry(inventory, keywords=keywords, roles=roles)
    if entry:
        return _spec_from_inventory_entry(entry)
    raise ValueError(f"Could not discover a control for keywords: {', '.join(keywords)}")


def _visible_inventory(inventory: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(entry) for entry in inventory if bool(entry.get("visible", True))]


def _entry_text_haystack(entry: dict[str, Any]) -> str:
    return " ".join(
        [
            str(entry.get("aria_label") or ""),
            str(entry.get("placeholder") or ""),
            str(entry.get("text") or ""),
            str(entry.get("name") or ""),
            str(entry.get("title") or ""),
            str(entry.get("id") or ""),
            str(entry.get("autocomplete") or ""),
            str(entry.get("type") or ""),
        ]
    ).lower()


def _entry_is_password(entry: dict[str, Any]) -> bool:
    entry_type = str(entry.get("type") or "").lower()
    autocomplete = str(entry.get("autocomplete") or "").lower()
    haystack = _entry_text_haystack(entry)
    return entry_type == "password" or "current-password" in autocomplete or "password" in autocomplete or "password" in haystack


def _entry_is_text_like(entry: dict[str, Any]) -> bool:
    entry_type = str(entry.get("type") or "").lower()
    tag = str(entry.get("tag") or "").lower()
    role = str(entry.get("role") or "").lower()
    autocomplete = str(entry.get("autocomplete") or "").lower()
    if _entry_is_password(entry):
        return False
    if tag in {"textarea"} or role in {"textbox", "combobox"}:
        return True
    if tag != "input":
        return False
    if entry_type in {"text", "email", "search", "tel", "url", "number", "date"}:
        return True
    if "username" in autocomplete or "email" in autocomplete:
        return True
    return False


def _entry_is_submit_capable(entry: dict[str, Any]) -> bool:
    tag = str(entry.get("tag") or "").lower()
    entry_type = str(entry.get("type") or "").lower()
    role = str(entry.get("role") or "").lower()
    return (
        role in {"button", "link"}
        or tag in {"button", "a"}
        or entry_type in {"submit", "button"}
    )


def _entry_matches_keywords(entry: dict[str, Any], keywords: Sequence[str]) -> bool:
    haystack = _entry_text_haystack(entry)
    return any(keyword.lower() in haystack for keyword in keywords)


def _entry_as_locator_spec(entry: dict[str, Any]) -> LocatorSpec:
    frame_name = entry.get("frame_name")
    frame_url = entry.get("frame_url")
    tag = str(entry.get("tag") or "").lower()
    entry_type = str(entry.get("type") or "").lower()
    autocomplete = str(entry.get("autocomplete") or "").lower()
    if entry.get("id"):
        return LocatorSpec(kind="css", selector=f"#{entry['id']}", frame_name=frame_name, frame_url=frame_url)
    if tag == "input" and entry_type == "email":
        return LocatorSpec(kind="css", selector='input[type="email"]', frame_name=frame_name, frame_url=frame_url)
    if tag == "input" and entry_type == "password":
        return LocatorSpec(kind="css", selector='input[type="password"]', frame_name=frame_name, frame_url=frame_url)
    if tag == "input" and entry_type == "submit":
        return LocatorSpec(kind="css", selector='input[type="submit"]', frame_name=frame_name, frame_url=frame_url)
    if tag == "button" and entry_type == "submit":
        return LocatorSpec(kind="css", selector='button[type="submit"]', frame_name=frame_name, frame_url=frame_url)
    if tag == "input" and ("username" in autocomplete or "email" in autocomplete):
        return LocatorSpec(kind="css", selector='input[autocomplete*="username"], input[autocomplete*="email"]', frame_name=frame_name, frame_url=frame_url)
    if entry.get("name") and tag == "input":
        return LocatorSpec(kind="css", selector=f'input[name="{entry["name"]}"]', frame_name=frame_name, frame_url=frame_url)
    if entry.get("placeholder"):
        return LocatorSpec(kind="placeholder", selector=str(entry["placeholder"]), exact=True, frame_name=frame_name, frame_url=frame_url)
    if entry.get("aria_label"):
        return LocatorSpec(kind="label", selector=str(entry["aria_label"]), exact=True, frame_name=frame_name, frame_url=frame_url)
    role = str(entry.get("role") or "").strip()
    text = _normalize_text(entry.get("text") or entry.get("name") or entry.get("title") or entry.get("id"))
    if role and text:
        return LocatorSpec(kind="role", role=role, name=text, exact=True, frame_name=frame_name, frame_url=frame_url)
    if text:
        return LocatorSpec(kind="text", selector=text, exact=True, frame_name=frame_name, frame_url=frame_url)
    if tag == "button":
        return LocatorSpec(kind="css", selector="button", frame_name=frame_name, frame_url=frame_url)
    raise ValueError(f"Could not derive a stable selector from control entry: {entry}")


def _fallback_single_visible_textbox(inventory: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    textboxes = [entry for entry in _visible_inventory(inventory) if _entry_is_text_like(entry)]
    password_entries = [entry for entry in _visible_inventory(inventory) if _entry_is_password(entry)]
    if len(textboxes) == 1 and password_entries:
        return textboxes[0]
    return None


def _fallback_single_submit_control(inventory: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    submit_controls = [entry for entry in _visible_inventory(inventory) if _entry_is_submit_capable(entry)]
    if len(submit_controls) == 1:
        return submit_controls[0]
    return None


def _discover_login_email_entry(snapshot: dict[str, Any] | None, inventory: Sequence[dict[str, Any]]) -> dict[str, Any]:
    keywords = ("email", "username", "user", "login", "identifier")
    entry = _find_inventory_entry(inventory, keywords=keywords, roles=("textbox", "combobox"))
    if entry:
        return entry
    visible_entries = _visible_inventory(inventory)
    for candidate in visible_entries:
        if not _entry_is_text_like(candidate):
            continue
        entry_type = str(candidate.get("type") or "").lower()
        autocomplete = str(candidate.get("autocomplete") or "").lower()
        haystack = _entry_text_haystack(candidate)
        if entry_type == "email" or "email" in autocomplete or "username" in autocomplete:
            return candidate
        if any(term in haystack for term in keywords):
            return candidate
    fallback = _fallback_single_visible_textbox(inventory)
    if fallback:
        return fallback
    if snapshot:
        entry = _find_node(snapshot, roles=("textbox", "combobox"), keywords=keywords)
        if entry:
            return entry
    raise ValueError("Could not discover Tandem login email/username control")


def _discover_login_password_entry(snapshot: dict[str, Any] | None, inventory: Sequence[dict[str, Any]]) -> dict[str, Any]:
    keywords = ("password",)
    entry = _find_inventory_entry(inventory, keywords=keywords, roles=("textbox", "combobox"))
    if entry:
        return entry
    for candidate in _visible_inventory(inventory):
        if _entry_is_password(candidate):
            return candidate
    if snapshot:
        entry = _find_node(snapshot, roles=("textbox", "combobox"), keywords=keywords)
        if entry:
            return entry
    raise ValueError("Could not discover Tandem login password control")


def _discover_login_submit_entry(snapshot: dict[str, Any] | None, inventory: Sequence[dict[str, Any]]) -> dict[str, Any]:
    keywords = ("sign in", "log in", "login", "continue", "next", "submit")
    entry = _find_inventory_entry(inventory, keywords=keywords, roles=("button", "link"))
    if entry:
        return entry
    for candidate in _visible_inventory(inventory):
        if _entry_is_submit_capable(candidate) and _entry_matches_keywords(candidate, keywords):
            return candidate
    fallback = _fallback_single_submit_control(inventory)
    if fallback:
        return fallback
    if snapshot:
        entry = _find_node(snapshot, roles=("button", "link"), keywords=keywords)
        if entry:
            return entry
    raise ValueError("Could not discover Tandem login submit control")


def discover_login_controls_from_controls(
    *,
    accessibility_snapshot: dict[str, Any] | None,
    control_inventory: Sequence[dict[str, Any]],
) -> tuple[LocatorSpec, LocatorSpec, LocatorSpec]:
    return (
        _entry_as_locator_spec(_discover_login_email_entry(accessibility_snapshot, control_inventory)),
        _entry_as_locator_spec(_discover_login_password_entry(accessibility_snapshot, control_inventory)),
        _entry_as_locator_spec(_discover_login_submit_entry(accessibility_snapshot, control_inventory)),
    )


def discover_timeline_controls_from_controls(
    *,
    accessibility_snapshot: dict[str, Any] | None,
    control_inventory: Sequence[dict[str, Any]],
) -> tuple[LocatorSpec, LocatorSpec, LocatorSpec, LocatorSpec]:
    return (
        _discover_spec(
            snapshot=accessibility_snapshot,
            inventory=control_inventory,
            roles=("button", "link"),
            keywords=("daily timeline", "timeline"),
        ),
        _discover_spec(
            snapshot=accessibility_snapshot,
            inventory=control_inventory,
            roles=("textbox", "spinbutton", "combobox"),
            keywords=("start", "from"),
        ),
        _discover_spec(
            snapshot=accessibility_snapshot,
            inventory=control_inventory,
            roles=("textbox", "spinbutton", "combobox"),
            keywords=("end", "to"),
        ),
        _discover_spec(
            snapshot=accessibility_snapshot,
            inventory=control_inventory,
            roles=("button", "link"),
            keywords=("export csv", "csv", "download csv"),
        ),
    )


def discover_tandem_page_map_from_controls(
    *,
    accessibility_snapshot: dict[str, Any] | None,
    control_inventory: Sequence[dict[str, Any]],
    login_url: str = DEFAULT_LOGIN_URL,
    daily_timeline_url: str | None = None,
    generated_at: str | None = None,
) -> TandemPageMap:
    login_email = _discover_spec(
        snapshot=accessibility_snapshot,
        inventory=control_inventory,
        roles=("textbox", "combobox"),
        keywords=("email", "username", "user"),
    )
    login_password = _discover_spec(
        snapshot=accessibility_snapshot,
        inventory=control_inventory,
        roles=("textbox", "combobox"),
        keywords=("password",),
    )
    login_submit = _discover_spec(
        snapshot=accessibility_snapshot,
        inventory=control_inventory,
        roles=("button", "link"),
        keywords=("sign in", "log in", "login", "continue"),
    )
    daily_timeline_nav = _discover_spec(
        snapshot=accessibility_snapshot,
        inventory=control_inventory,
        roles=("button", "link"),
        keywords=("daily timeline", "timeline"),
    )
    start_date = _discover_spec(
        snapshot=accessibility_snapshot,
        inventory=control_inventory,
        roles=("textbox", "spinbutton", "combobox"),
        keywords=("start", "from"),
    )
    end_date = _discover_spec(
        snapshot=accessibility_snapshot,
        inventory=control_inventory,
        roles=("textbox", "spinbutton", "combobox"),
        keywords=("end", "to"),
    )
    export_csv = _discover_spec(
        snapshot=accessibility_snapshot,
        inventory=control_inventory,
        roles=("button", "link"),
        keywords=("export csv", "csv", "download csv"),
    )
    return TandemPageMap(
        login_url=login_url,
        daily_timeline_url=daily_timeline_url,
        login_email=login_email,
        login_password=login_password,
        login_submit=login_submit,
        daily_timeline_nav=daily_timeline_nav,
        start_date=start_date,
        end_date=end_date,
        export_csv=export_csv,
        generated_at=generated_at or datetime.utcnow().isoformat(timespec="seconds") + "Z",
        source="playwright-discovery",
    )


def discover_tandem_page_map(page, *, login_url: str = DEFAULT_LOGIN_URL, daily_timeline_url: str | None = None) -> TandemPageMap:
    snapshot = _capture_accessibility_snapshot(page)
    inventory = capture_control_inventory(page)
    return discover_tandem_page_map_from_controls(
        accessibility_snapshot=snapshot,
        control_inventory=inventory,
        login_url=login_url,
        daily_timeline_url=daily_timeline_url,
    )
