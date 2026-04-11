"""
tests/unit/test_deprecations.py
================================
Phase E2 — Deprecation governance tests.

Validates:
1. deprecations.yaml file structure and schema
2. Each entry has required fields with valid formats
3. No duplicate ids or targets
4. Each module-kind shim emits DeprecationWarning on import
5. Each module shim warning mentions the new path token
6. Each module shim warning mentions the v4.2 removal version
"""
from __future__ import annotations

import importlib
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import pytest
import yaml

# ── Load deprecations.yaml ───────────────────────────────────────────────────
REPO_ROOT  = Path(__file__).parents[2]
DEP_FILE   = REPO_ROOT / "deprecations.yaml"

_dep_data: dict[str, Any] = {}
_entries: list[dict] = []

if DEP_FILE.exists():
    _dep_data = yaml.safe_load(DEP_FILE.read_text(encoding="utf-8")) or {}
    _entries  = _dep_data.get("deprecations", [])

_MODULE_ENTRIES = [e for e in _entries if e.get("kind") == "module"]

_REQUIRED_FIELDS = {
    "id", "kind", "target", "replacement",
    "warn_since", "remove_in", "migration_note",
}
_VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. File-level structure
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeprecationsFileStructure:

    def test_file_exists(self):
        assert DEP_FILE.exists(), f"deprecations.yaml not found at {DEP_FILE}"

    def test_schema_version_present(self):
        assert "schema_version" in _dep_data

    def test_generated_field_present(self):
        assert "generated" in _dep_data

    def test_deprecations_list_present(self):
        assert isinstance(_entries, list) and len(_entries) >= 1

    def test_has_module_entries(self):
        assert len(_MODULE_ENTRIES) >= 6, "Expected at least 6 module deprecation entries"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Per-entry structure
# ═══════════════════════════════════════════════════════════════════════════════

class TestEntryStructure:

    @pytest.mark.parametrize("entry", _entries, ids=[e.get("id","?") for e in _entries])
    def test_required_fields_present(self, entry):
        missing = _REQUIRED_FIELDS - set(entry.keys())
        assert not missing, f"Entry {entry.get('id')!r} missing: {missing}"

    @pytest.mark.parametrize("entry", _entries, ids=[e.get("id","?") for e in _entries])
    def test_warn_since_semver(self, entry):
        v = str(entry.get("warn_since", ""))
        assert _VERSION_RE.match(v), f"{entry.get('id')!r} warn_since={v!r} not X.Y.Z"

    @pytest.mark.parametrize("entry", _entries, ids=[e.get("id","?") for e in _entries])
    def test_remove_in_semver(self, entry):
        v = str(entry.get("remove_in", ""))
        assert _VERSION_RE.match(v), f"{entry.get('id')!r} remove_in={v!r} not X.Y.Z"

    @pytest.mark.parametrize("entry", _entries, ids=[e.get("id","?") for e in _entries])
    def test_remove_in_gt_warn_since(self, entry):
        from packaging.version import Version
        ws = Version(str(entry.get("warn_since", "0.0.0")))
        ri = Version(str(entry.get("remove_in",  "0.0.0")))
        assert ri > ws, f"{entry.get('id')!r}: remove_in {ri} must be > warn_since {ws}"

    @pytest.mark.parametrize("entry", _entries, ids=[e.get("id","?") for e in _entries])
    def test_kind_valid(self, entry):
        valid = {"module", "script", "function", "class", "parameter"}
        assert entry.get("kind") in valid, f"{entry.get('id')!r} invalid kind={entry.get('kind')!r}"

    def test_no_duplicate_ids(self):
        ids = [e.get("id") for e in _entries]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {[i for i in ids if ids.count(i)>1]}"

    def test_no_duplicate_targets(self):
        targets = [e.get("target") for e in _entries]
        dupes = [t for t in targets if targets.count(t) > 1]
        assert not dupes, f"Duplicate targets: {dupes}"

    @pytest.mark.parametrize("entry", _MODULE_ENTRIES, ids=[e.get("id","?") for e in _MODULE_ENTRIES])
    def test_module_entry_has_warn_template(self, entry):
        assert "warn_template" in entry, f"Module entry {entry.get('id')!r} missing warn_template"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Shim import + DeprecationWarning
# ═══════════════════════════════════════════════════════════════════════════════

def _import_fresh(module_name: str) -> list:
    """Import a module fresh (evicting from sys.modules) and capture all warnings."""
    # Evict the module and any child modules cached from a previous import
    to_evict = [k for k in sys.modules if k == module_name or k.startswith(module_name + ".")]
    for k in to_evict:
        del sys.modules[k]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            pytest.skip(f"Module {module_name!r} not importable: {exc}")
    return list(caught)


class TestModuleShimWarnings:
    """Each module-kind shim must emit DeprecationWarning with the right tokens."""

    @pytest.mark.parametrize(
        "entry", _MODULE_ENTRIES,
        ids=[e.get("id", "?") for e in _MODULE_ENTRIES],
    )
    def test_shim_emits_deprecation_warning(self, entry):
        caught = _import_fresh(entry["target"])
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert dep, (
            f"Importing {entry['target']!r} did not emit DeprecationWarning.\n"
            f"  All warnings captured: {[str(w.message) for w in caught]}"
        )

    @pytest.mark.parametrize(
        "entry", _MODULE_ENTRIES,
        ids=[e.get("id", "?") for e in _MODULE_ENTRIES],
    )
    def test_shim_warns_new_path(self, entry):
        """Warning text must contain the new_path_token from warn_template."""
        token = entry["warn_template"]["new_path_token"]
        caught = _import_fresh(entry["target"])
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        if not dep:
            pytest.skip("No DeprecationWarning (covered by previous test)")
        text = " ".join(str(w.message) for w in dep)
        assert token in text, (
            f"Shim {entry['target']!r} warning does not mention new path token {token!r}.\n"
            f"  Warning text: {text!r}"
        )

    @pytest.mark.parametrize(
        "entry", _MODULE_ENTRIES,
        ids=[e.get("id", "?") for e in _MODULE_ENTRIES],
    )
    def test_shim_warns_removal_version(self, entry):
        """Warning text must contain the remove_version_token from warn_template."""
        token = entry["warn_template"]["remove_version_token"]   # e.g. "v4.2"
        caught = _import_fresh(entry["target"])
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        if not dep:
            pytest.skip("No DeprecationWarning (covered by previous test)")
        text = " ".join(str(w.message) for w in dep)
        assert token in text, (
            f"Shim {entry['target']!r} warning does not mention removal version {token!r}.\n"
            f"  Warning text: {text!r}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. All shims must be importable without crashing
# ═══════════════════════════════════════════════════════════════════════════════

class TestShimImportability:

    @pytest.mark.parametrize(
        "entry", _MODULE_ENTRIES,
        ids=[e.get("id", "?") for e in _MODULE_ENTRIES],
    )
    def test_shim_importable_no_error(self, entry):
        target = entry["target"]
        to_evict = [k for k in sys.modules if k == target or k.startswith(target + ".")]
        for k in to_evict:
            del sys.modules[k]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            try:
                importlib.import_module(target)
            except ImportError as exc:
                pytest.fail(f"Shim {target!r} raised ImportError: {exc}")
