"""
Scope System — Privacy & Access Control
==========================================

Four-level privacy scope model ported from ErnOS V4 (src/core/scopes.py).

Every piece of state in NeuroForm is tagged with a scope from creation.
The access matrix enforces hard data-flow boundaries:
  - CORE_PRIVATE: internal only, data enters but never leaves
  - CORE_PUBLIC: system knowledge, informs all forward layers
  - PRIVATE: DM engagement, never crosses to PUBLIC
  - PUBLIC: public channel, accessible by all layers
"""
import logging
from enum import IntEnum
from typing import Optional

logger = logging.getLogger(__name__)


class Scope(IntEnum):
    """Privacy scopes ordered by privilege level."""
    CORE_PRIVATE = 1   # Internal only (safety, autobiography)
    CORE_PUBLIC = 2    # System knowledge (shared, read-only for users)
    PRIVATE = 3        # User DM — never crosses to PUBLIC
    PUBLIC = 4         # Public channels — accessible by all


# ── Access Control Matrix ──────────────────────────────────────────
#
# Who can READ whose data:
#   CORE_PRIVATE → can read ALL scopes
#   CORE_PUBLIC  → can read CORE_PUBLIC + PUBLIC
#   PRIVATE      → can read PRIVATE + PUBLIC + CORE_PUBLIC
#   PUBLIC       → can read PUBLIC + CORE_PUBLIC

_ACCESS_MATRIX = {
    Scope.CORE_PRIVATE: frozenset({
        Scope.CORE_PRIVATE, Scope.CORE_PUBLIC,
        Scope.PRIVATE, Scope.PUBLIC,
    }),
    Scope.CORE_PUBLIC: frozenset({
        Scope.CORE_PUBLIC, Scope.PUBLIC,
    }),
    Scope.PRIVATE: frozenset({
        Scope.PRIVATE, Scope.PUBLIC, Scope.CORE_PUBLIC,
    }),
    Scope.PUBLIC: frozenset({
        Scope.PUBLIC, Scope.CORE_PUBLIC,
    }),
}


def check_access(request_scope: Scope, resource_scope: Scope) -> bool:
    """Check if request_scope has permission to read resource_scope."""
    allowed = _ACCESS_MATRIX.get(request_scope, frozenset())
    return resource_scope in allowed


def resolve_scope(*, is_dm: bool = False, is_system: bool = False) -> Scope:
    """
    Determine scope from platform context.

    - System/internal calls → CORE_PRIVATE
    - DM messages → PRIVATE
    - Everything else → PUBLIC
    """
    if is_system:
        return Scope.CORE_PRIVATE
    return Scope.PRIVATE if is_dm else Scope.PUBLIC


class ScopeManager:
    """
    Manages scope-aware filtering across all memory tiers.

    Wraps access checks and provides scope resolution for the
    bridge, orchestrator, and individual memory modules.
    """

    def __init__(self, enable_scopes: bool = True):
        self.enable_scopes = enable_scopes
        logger.info(f"ScopeManager initialized (scopes={'enabled' if enable_scopes else 'disabled'})")

    def check(self, request_scope: Scope, resource_scope: Scope) -> bool:
        """Check access with optional global disable."""
        if not self.enable_scopes:
            return True
        return check_access(request_scope, resource_scope)

    def resolve(self, *, is_dm: bool = False, is_system: bool = False) -> Scope:
        """Resolve scope from context. Returns PUBLIC if scopes disabled."""
        if not self.enable_scopes:
            return Scope.PUBLIC
        return resolve_scope(is_dm=is_dm, is_system=is_system)

    def filter_by_scope(
        self,
        items: list,
        request_scope: Scope,
        scope_key: str = "scope",
    ) -> list:
        """
        Filter a list of dicts/objects by scope access rules.

        Each item should have a 'scope' key (str or Scope).
        Returns only items the request_scope is allowed to see.
        """
        if not self.enable_scopes:
            return items

        result = []
        for item in items:
            # Extract scope from item
            if isinstance(item, dict):
                raw_scope = item.get(scope_key, "PUBLIC")
            elif hasattr(item, scope_key):
                raw_scope = getattr(item, scope_key)
            else:
                raw_scope = "PUBLIC"

            # Convert string to Scope enum
            if isinstance(raw_scope, str):
                try:
                    resource_scope = Scope[raw_scope.upper()]
                except KeyError:
                    resource_scope = Scope.PUBLIC
            else:
                resource_scope = raw_scope

            if self.check(request_scope, resource_scope):
                result.append(item)

        return result

    def snapshot(self) -> dict:
        """Diagnostic snapshot."""
        return {
            "enabled": self.enable_scopes,
            "scopes": [s.name for s in Scope],
        }
