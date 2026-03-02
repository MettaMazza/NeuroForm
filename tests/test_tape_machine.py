"""Tests for neuroform.memory.tape_machine — Cell, CognitiveTape, TapeMachine."""
import json
import os
import pytest
from neuroform.memory.tape_machine import (
    Cell, CognitiveTape, TapeMachine, TapeFaultError,
)


class TestCell:
    def test_cell_creation(self):
        c = Cell(type="MEMORY", content="hello", readonly=False, timestamp=1.0)
        assert c.type == "MEMORY"
        assert c.content == "hello"
        assert c.readonly is False
        assert c.timestamp == 1.0

    def test_cell_readonly(self):
        c = Cell(type="KERNEL", content="core", readonly=True, timestamp=1.0)
        assert c.readonly is True


class TestCognitiveTape:
    @pytest.fixture
    def tape(self):
        return CognitiveTape()

    def test_write_and_read(self, tape):
        tape.write((0, 0, 0), "MEMORY", "hello")
        cell = tape.read((0, 0, 0))
        assert cell is not None
        assert cell.content == "hello"

    def test_write_readonly_blocks_overwrite(self, tape):
        tape.write((0, 0, 0), "KERNEL", "locked", readonly=True)
        with pytest.raises(TapeFaultError, match="READ-ONLY"):
            tape.write((0, 0, 0), "MEMORY", "attempt")

    def test_update_existing(self, tape):
        tape.write((0, 0, 0), "MEMORY", "v1")
        tape.update((0, 0, 0), "v2")
        assert tape.read((0, 0, 0)).content == "v2"

    def test_update_missing_raises(self, tape):
        with pytest.raises(TapeFaultError, match="No cell"):
            tape.update((5, 5, 5), "content")

    def test_update_readonly_raises(self, tape):
        tape.write((0, 0, 0), "KERNEL", "locked", readonly=True)
        with pytest.raises(TapeFaultError, match="READ-ONLY"):
            tape.update((0, 0, 0), "new")

    def test_delete(self, tape):
        tape.write((0, 0, 0), "MEMORY", "temp")
        tape.delete((0, 0, 0))
        assert tape.read((0, 0, 0)) is None

    def test_delete_empty_raises(self, tape):
        with pytest.raises(TapeFaultError, match="empty"):
            tape.delete((1, 1, 1))

    def test_delete_readonly_raises(self, tape):
        tape.write((0, 0, 0), "KERNEL", "locked", readonly=True)
        with pytest.raises(TapeFaultError, match="READ-ONLY"):
            tape.delete((0, 0, 0))

    def test_len(self, tape):
        assert len(tape) == 0
        tape.write((0, 0, 0), "MEMORY", "a")
        tape.write((1, 0, 0), "MEMORY", "b")
        assert len(tape) == 2

    def test_read_empty_returns_none(self, tape):
        assert tape.read((99, 99, 99)) is None

    def test_to_dict_and_from_dict(self, tape):
        tape.write((0, 0, 0), "KERNEL", "core", readonly=True)
        tape.write((1, 2, 3), "MEMORY", "data")
        d = tape.to_dict()
        tape2 = CognitiveTape.from_dict(d)
        assert len(tape2) == 2
        assert tape2.read((0, 0, 0)).content == "core"
        assert tape2.read((0, 0, 0)).readonly is True
        assert tape2.read((1, 2, 3)).content == "data"

    def test_from_dict_malformed_coords(self):
        d = {"bad_key": {"type": "M", "content": "x", "readonly": False, "timestamp": 1.0}}
        tape = CognitiveTape.from_dict(d)
        assert len(tape) == 0  # Skipped

    def test_from_dict_empty(self):
        tape = CognitiveTape.from_dict({})
        assert len(tape) == 0

    def test_from_dict_non_dict(self):
        tape = CognitiveTape.from_dict("not a dict")
        assert len(tape) == 0


class TestTapeMachine:
    @pytest.fixture
    def tm(self, tmp_path):
        return TapeMachine(user_id="test_user", persist_dir=str(tmp_path))

    def test_init_creates_kernel(self, tm):
        assert len(tm.tape) >= 1
        cell = tm.tape.read((0, 2, 0))
        assert cell is not None
        assert cell.type == "KERNEL"
        assert cell.readonly is True

    def test_seek(self, tm):
        tm.op_seek((5, 3, 2))
        assert tm.focus_pointer == (5, 3, 2)

    def test_move_up(self, tm):
        tm.op_seek((0, 0, 0))
        tm.op_move("UP")
        assert tm.focus_pointer == (0, 1, 0)

    def test_move_down(self, tm):
        tm.op_seek((0, 2, 0))
        tm.op_move("DOWN")
        assert tm.focus_pointer == (0, 1, 0)

    def test_move_down_clamped(self, tm):
        tm.op_seek((0, 0, 0))
        tm.op_move("DOWN")
        assert tm.focus_pointer == (0, 0, 0)

    def test_move_in(self, tm):
        tm.op_seek((0, 0, 0))
        tm.op_move("IN")
        assert tm.focus_pointer == (0, 0, 1)

    def test_move_out(self, tm):
        tm.op_seek((0, 0, 3))
        tm.op_move("OUT")
        assert tm.focus_pointer == (0, 0, 2)

    def test_move_out_clamped(self, tm):
        tm.op_seek((0, 0, 0))
        tm.op_move("OUT")
        assert tm.focus_pointer == (0, 0, 0)

    def test_move_invalid_direction(self, tm):
        with pytest.raises(TapeFaultError, match="Unknown direction"):
            tm.op_move("LEFT")

    def test_read_empty(self, tm):
        tm.op_seek((99, 99, 99))
        assert tm.op_read() == ""

    def test_read_existing(self, tm):
        tm.op_seek((0, 2, 0))
        content = tm.op_read()
        assert "NeuroForm" in content

    def test_write_new_cell(self, tm):
        tm.op_seek((10, 0, 0))
        tm.op_write("new data")
        assert tm.op_read() == "new data"

    def test_write_overwrites_existing(self, tm):
        tm.op_seek((10, 0, 0))
        tm.op_write("v1")
        tm.op_write("v2")
        assert tm.op_read() == "v2"

    def test_write_identity_triggers_snapshot(self, tm):
        tm.tape.write((5, 0, 0), "IDENTITY", "old identity")
        tm.op_seek((5, 0, 0))
        tm.op_write("new identity")  # Should trigger snapshot
        assert tm.op_read() == "new identity"

    def test_insert_empty_cell(self, tm):
        tm.op_seek((20, 0, 0))
        tm.op_insert("SCRATCHPAD", "notes")
        assert tm.tape.read((20, 0, 0)).type == "SCRATCHPAD"
        assert tm.tape.read((20, 0, 0)).content == "notes"

    def test_insert_shifts_existing(self, tm):
        tm.op_seek((0, 0, 0))
        tm.op_write("cell_0")
        tm.op_seek((1, 0, 0))
        tm.op_write("cell_1")
        # Insert at 0 — should shift cell_0 to (1,0,0) and cell_1 to (2,0,0)
        tm.op_seek((0, 0, 0))
        tm.op_insert("NEW", "inserted")
        assert tm.tape.read((0, 0, 0)).content == "inserted"
        assert tm.tape.read((1, 0, 0)).content == "cell_0"
        assert tm.tape.read((2, 0, 0)).content == "cell_1"

    def test_delete_existing(self, tm):
        tm.op_seek((10, 0, 0))
        tm.op_write("temp")
        tm.op_delete()
        assert tm.op_read() == ""

    def test_delete_empty_raises(self, tm):
        tm.op_seek((99, 99, 99))
        with pytest.raises(TapeFaultError, match="empty"):
            tm.op_delete()

    def test_delete_identity_snapshots(self, tm):
        tm.tape.write((5, 0, 0), "IDENTITY", "identity cell")
        tm.op_seek((5, 0, 0))
        tm.op_delete()
        assert tm.tape.read((5, 0, 0)) is None

    def test_scan_finds_content(self, tm):
        tm.op_seek((10, 0, 0))
        tm.op_write("important data about cats")
        tm.op_seek((0, 0, 0))  # Move away
        result = tm.op_scan("cats")
        assert result is True
        assert tm.focus_pointer == (10, 0, 0)

    def test_scan_finds_by_type(self, tm):
        result = tm.op_scan("KERNEL")
        assert result is True
        assert tm.tape.read(tm.focus_pointer).type == "KERNEL"

    def test_scan_not_found_raises(self, tm):
        with pytest.raises(TapeFaultError, match="not found"):
            tm.op_scan("nonexistent_content_xyz")

    def test_scan_empty_tape_raises(self, tmp_path):
        tm = TapeMachine(user_id="empty_user", persist_dir=str(tmp_path))
        tm.tape.cells.clear()
        with pytest.raises(TapeFaultError, match="empty"):
            tm.op_scan("anything")

    def test_get_view(self, tm):
        view = tm.get_view()
        assert "TAPE MACHINE" in view
        assert "Total:" in view

    def test_get_view_blank_region(self, tm):
        tm.op_seek((50, 50, 50))
        view = tm.get_view()
        assert "BLANK" in view

    def test_get_index(self, tm):
        idx = tm.get_index()
        assert "TAPE INDEX" in idx
        assert "KERNEL" in idx

    def test_get_index_empty(self, tmp_path):
        tm = TapeMachine(user_id="empty_idx", persist_dir=str(tmp_path))
        tm.tape.cells.clear()
        idx = tm.get_index()
        assert "Empty" in idx

    def test_save_and_load(self, tmp_path):
        tm = TapeMachine(user_id="persist", persist_dir=str(tmp_path))
        tm.op_seek((5, 0, 0))
        tm.op_write("persistent data")
        tm.save()

        tm2 = TapeMachine(user_id="persist", persist_dir=str(tmp_path))
        tm2.op_seek((5, 0, 0))
        assert tm2.op_read() == "persistent data"

    def test_snapshot_and_restore(self, tmp_path):
        tm = TapeMachine(user_id="snap", persist_dir=str(tmp_path))
        tm.op_seek((5, 0, 0))
        tm.op_write("before snapshot")
        tm.create_snapshot()
        tm.op_write("after snapshot")
        tm._restore_from_backup()
        tm.op_seek((5, 0, 0))
        assert tm.op_read() == "before snapshot"

    def test_restore_no_backup(self, tmp_path):
        tm = TapeMachine(user_id="no_bak", persist_dir=str(tmp_path))
        tm._restore_from_backup()  # Should not crash
        assert len(tm.tape) == 0

    def test_load_corrupt_file(self, tmp_path):
        user_dir = tmp_path / "corrupt"
        user_dir.mkdir()
        tape_file = user_dir / "tape.json"
        tape_file.write_text("not valid json")
        tm = TapeMachine(user_id="corrupt", persist_dir=str(tmp_path))
        # Should handle error and start fresh
        assert isinstance(tm.tape, CognitiveTape)

    def test_save_error(self, tm):
        tm.tape_path = "/dev/null/impossible/tape.json"
        tm.save()  # Should not crash

    def test_snapshot_diagnostic(self, tm):
        snap = tm.snapshot()
        assert snap["user_id"] == "test_user"
        assert snap["scope"] == "PUBLIC"
        assert "cell_count" in snap
        assert "focus_pointer" in snap

    def test_scope_private_path(self, tmp_path):
        tm = TapeMachine(user_id="user1", scope="PRIVATE",
                         persist_dir=str(tmp_path))
        assert "private" in tm.tape_path

    def test_get_view_with_content(self, tm):
        tm.op_seek((0, 0, 0))
        tm.op_write("turn 1")
        tm.op_seek((1, 0, 0))
        tm.op_write("turn 2")
        tm.op_seek((0, 0, 0))
        view = tm.get_view()
        assert "turn 1" in view
        assert ">>" in view  # Focus pointer marker

    def test_get_index_with_locked(self, tm):
        idx = tm.get_index()
        assert "[LOCKED]" in idx  # KERNEL cell is locked

    def test_get_view_long_content(self, tm):
        tm.op_seek((10, 0, 0))
        tm.op_write("x" * 300)  # Longer than 200 char preview
        tm.op_seek((10, 0, 0))
        view = tm.get_view()
        assert "..." in view

    def test_get_index_long_content(self, tm):
        tm.op_seek((10, 0, 0))
        tm.op_write("y" * 100)  # Longer than 80 char preview
        idx = tm.get_index()
        assert "..." in idx

    def test_scan_cross_z_levels(self, tm):
        tm.tape.write((0, 0, 5), "MEMORY", "deep thread data")
        tm.op_seek((0, 0, 0))
        result = tm.op_scan("deep thread")
        assert result is True
        assert tm.focus_pointer == (0, 0, 5)

    def test_create_snapshot_no_tape_file(self, tmp_path):
        tm = TapeMachine(user_id="no_snap", persist_dir=str(tmp_path))
        os.remove(tm.tape_path)  # Remove the tape file
        tm.create_snapshot()  # Should not crash (nothing to copy)

    def test_load_legacy_format(self, tmp_path):
        """Load tape file without focus_pointer (legacy)."""
        user_dir = tmp_path / "legacy"
        user_dir.mkdir()
        tape_file = user_dir / "tape.json"
        tape_file.write_text(json.dumps({
            "0,0,0": {"type": "MEMORY", "content": "legacy",
                      "readonly": False, "timestamp": 1.0}
        }))
        tm = TapeMachine(user_id="legacy", persist_dir=str(tmp_path))
        assert tm.tape.read((0, 0, 0)).content == "legacy"

    def test_from_dict_invalid_cell_data(self):
        """Cover L117-118: from_dict with cell data that causes TypeError."""
        d = {"0,0,0": {"type": "MEMORY", "content": 123}}  # Missing fields
        tape = CognitiveTape.from_dict(d)
        assert len(tape) == 0  # Skipped due to TypeError

    def test_snapshot_error(self, tmp_path):
        """Cover L192-193: create_snapshot error path."""
        tm = TapeMachine(user_id="snap_err", persist_dir=str(tmp_path))
        tm.backup_path = "/dev/null/impossible/tape.bak"
        tm.create_snapshot()  # Should not crash

    def test_restore_from_corrupt_backup(self, tmp_path):
        """Cover L207-211: backup restore with corrupt .bak file."""
        user_dir = tmp_path / "corrupt_bak"
        user_dir.mkdir()
        tape_file = user_dir / "tape.json"
        tape_file.write_text("invalid json")
        bak_file = user_dir / "tape.bak"
        bak_file.write_text("also invalid json")
        tm = TapeMachine(user_id="corrupt_bak", persist_dir=str(tmp_path))
        # Load fails, restore from backup also fails → empty tape
        assert isinstance(tm.tape, CognitiveTape)

    def test_restore_backup_legacy_format(self, tmp_path):
        """Cover L207: backup restore with legacy format (no cells key)."""
        user_dir = tmp_path / "bak_legacy"
        user_dir.mkdir()
        # Create a corrupt main tape
        tape_file = user_dir / "tape.json"
        tape_file.write_text("invalid json")
        # Create a valid legacy-format backup
        bak_file = user_dir / "tape.bak"
        bak_file.write_text(json.dumps({
            "0,0,0": {"type": "MEMORY", "content": "from_backup",
                      "readonly": False, "timestamp": 1.0}
        }))
        tm = TapeMachine(user_id="bak_legacy", persist_dir=str(tmp_path))
        assert tm.tape.read((0, 0, 0)).content == "from_backup"
