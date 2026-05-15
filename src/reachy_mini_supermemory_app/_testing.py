"""Test-only entry points re-exported from a single namespace.

Production modules carry reset hooks that exist solely so unit tests can wipe
module-level caches and singletons between cases. Importing those underscore-
prefixed names directly from production modules creates an implicit "test
helpers ARE part of the public surface" contract — refactors that rename or
move them silently break unrelated test files.

This module is the supported re-export point. Tests should import from here.
Production modules retain the underscore-prefixed originals so internal usage
keeps working; we just stop relying on tests reaching into them directly.
"""

from __future__ import annotations

from ._privacy_mode import _reset_for_tests as _reset_privacy_for_tests
from ._supermemory_client import (
    _reset_clients_for_tests as _reset_supermemory_clients,
)
from ._supermemory_client import (
    _reset_tag_cache_for_tests as _reset_supermemory_tag_cache,
)


def reset_supermemory_tag_cache() -> None:
    """Drop the discovered-containerTag cache."""
    _reset_supermemory_tag_cache()


def reset_supermemory_clients() -> None:
    """Drop the per-loop httpx.AsyncClient cache."""
    _reset_supermemory_clients()


def reset_privacy_state() -> None:
    """Clear the global privacy-active flag."""
    _reset_privacy_for_tests()


def reset_all() -> None:
    """Reset every module-level cache touched by the test fixtures.

    Convenience for autouse fixtures that don't need fine-grained control.
    """
    reset_supermemory_tag_cache()
    reset_supermemory_clients()
    reset_privacy_state()
