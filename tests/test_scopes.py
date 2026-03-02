"""Tests for neuroform.memory.scopes — Scope enum, access matrix, ScopeManager."""
import pytest
from neuroform.memory.scopes import Scope, check_access, resolve_scope, ScopeManager


class TestScopeEnum:
    def test_scope_values(self):
        assert Scope.CORE_PRIVATE == 1
        assert Scope.CORE_PUBLIC == 2
        assert Scope.PRIVATE == 3
        assert Scope.PUBLIC == 4

    def test_scope_ordering(self):
        assert Scope.CORE_PRIVATE < Scope.CORE_PUBLIC < Scope.PRIVATE < Scope.PUBLIC


class TestCheckAccess:
    def test_core_private_sees_all(self):
        for scope in Scope:
            assert check_access(Scope.CORE_PRIVATE, scope) is True

    def test_core_public_sees_public_and_core_public(self):
        assert check_access(Scope.CORE_PUBLIC, Scope.PUBLIC) is True
        assert check_access(Scope.CORE_PUBLIC, Scope.CORE_PUBLIC) is True
        assert check_access(Scope.CORE_PUBLIC, Scope.PRIVATE) is False
        assert check_access(Scope.CORE_PUBLIC, Scope.CORE_PRIVATE) is False

    def test_private_sees_private_public_core_public(self):
        assert check_access(Scope.PRIVATE, Scope.PRIVATE) is True
        assert check_access(Scope.PRIVATE, Scope.PUBLIC) is True
        assert check_access(Scope.PRIVATE, Scope.CORE_PUBLIC) is True
        assert check_access(Scope.PRIVATE, Scope.CORE_PRIVATE) is False

    def test_public_sees_public_and_core_public(self):
        assert check_access(Scope.PUBLIC, Scope.PUBLIC) is True
        assert check_access(Scope.PUBLIC, Scope.CORE_PUBLIC) is True
        assert check_access(Scope.PUBLIC, Scope.PRIVATE) is False
        assert check_access(Scope.PUBLIC, Scope.CORE_PRIVATE) is False

    def test_unknown_scope_returns_false(self):
        # Scope that doesn't exist in the matrix
        assert check_access(999, Scope.PUBLIC) is False


class TestResolveScope:
    def test_system_returns_core_private(self):
        assert resolve_scope(is_system=True) == Scope.CORE_PRIVATE

    def test_dm_returns_private(self):
        assert resolve_scope(is_dm=True) == Scope.PRIVATE

    def test_default_returns_public(self):
        assert resolve_scope() == Scope.PUBLIC

    def test_system_takes_precedence_over_dm(self):
        assert resolve_scope(is_dm=True, is_system=True) == Scope.CORE_PRIVATE


class TestScopeManager:
    @pytest.fixture
    def sm(self):
        return ScopeManager(enable_scopes=True)

    @pytest.fixture
    def sm_disabled(self):
        return ScopeManager(enable_scopes=False)

    def test_check_access_enabled(self, sm):
        assert sm.check(Scope.PUBLIC, Scope.PUBLIC) is True
        assert sm.check(Scope.PUBLIC, Scope.PRIVATE) is False

    def test_check_access_disabled_allows_all(self, sm_disabled):
        assert sm_disabled.check(Scope.PUBLIC, Scope.PRIVATE) is True
        assert sm_disabled.check(Scope.PUBLIC, Scope.CORE_PRIVATE) is True

    def test_resolve_enabled(self, sm):
        assert sm.resolve(is_dm=True) == Scope.PRIVATE
        assert sm.resolve(is_system=True) == Scope.CORE_PRIVATE

    def test_resolve_disabled_always_public(self, sm_disabled):
        assert sm_disabled.resolve(is_dm=True) == Scope.PUBLIC
        assert sm_disabled.resolve(is_system=True) == Scope.PUBLIC

    def test_filter_by_scope_dict_items(self, sm):
        items = [
            {"text": "public fact", "scope": "PUBLIC"},
            {"text": "private fact", "scope": "PRIVATE"},
            {"text": "core fact", "scope": "CORE_PRIVATE"},
        ]
        result = sm.filter_by_scope(items, Scope.PUBLIC)
        assert len(result) == 1
        assert result[0]["text"] == "public fact"

    def test_filter_by_scope_private_sees_more(self, sm):
        items = [
            {"text": "public", "scope": "PUBLIC"},
            {"text": "private", "scope": "PRIVATE"},
        ]
        result = sm.filter_by_scope(items, Scope.PRIVATE)
        assert len(result) == 2

    def test_filter_by_scope_disabled(self, sm_disabled):
        items = [
            {"text": "core", "scope": "CORE_PRIVATE"},
            {"text": "private", "scope": "PRIVATE"},
        ]
        result = sm_disabled.filter_by_scope(items, Scope.PUBLIC)
        assert len(result) == 2  # Disabled → all pass

    def test_filter_by_scope_object_items(self, sm):
        class Item:
            def __init__(self, scope):
                self.scope = scope
        items = [Item("PUBLIC"), Item("PRIVATE")]
        result = sm.filter_by_scope(items, Scope.PUBLIC)
        assert len(result) == 1

    def test_filter_by_scope_unknown_scope_string(self, sm):
        items = [{"text": "x", "scope": "UNKNOWN_SCOPE"}]
        result = sm.filter_by_scope(items, Scope.PUBLIC)
        # Unknown scope defaults to PUBLIC
        assert len(result) == 1

    def test_filter_by_scope_no_scope_key(self, sm):
        items = [{"text": "no scope key"}]
        result = sm.filter_by_scope(items, Scope.PUBLIC)
        # Missing scope defaults to PUBLIC
        assert len(result) == 1

    def test_filter_by_scope_enum_value(self, sm):
        items = [{"text": "x", "scope": Scope.PRIVATE}]
        result = sm.filter_by_scope(items, Scope.PUBLIC)
        assert len(result) == 0

    def test_snapshot(self, sm):
        snap = sm.snapshot()
        assert snap["enabled"] is True
        assert "CORE_PRIVATE" in snap["scopes"]
        assert "PUBLIC" in snap["scopes"]

    def test_filter_by_scope_no_attr_fallback(self, sm):
        """L120: item that's not dict and has no 'scope' attribute → defaults to PUBLIC."""
        items = [42, "plain_string"]
        result = sm.filter_by_scope(items, Scope.PUBLIC)
        assert len(result) == 2  # Both default to PUBLIC, PUBLIC can read PUBLIC
