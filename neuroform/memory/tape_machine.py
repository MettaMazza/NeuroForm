"""
Tier 5: Tape Machine — 3D Cognitive Computation Tape
======================================================

3D spatial memory structure ported from ErnOS V3/V4
(src/memory/tape_machine.py, src/tape/tape.py).

NOT a simple memory store — this is the AI's personal Turing-complete
computation structure. Cells are keyed by (x, y, z) coordinates:
  - X axis: Linear time / sequence
  - Y axis: Abstraction level (0=surface, 2=deep)
  - Z axis: Thread / parallel computation

Features:
  - Focus pointer navigation (seek, move)
  - Read/write/insert/delete with readonly enforcement
  - Z-level-aware scan search
  - Scope-aware per-user persistence
  - Snapshot/restore for safety
"""
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Optional, Any

logger = logging.getLogger(__name__)

Coord = Tuple[int, int, int]

PERSIST_DIR = os.path.join(os.getcwd(), "memory", "tape")


class TapeFaultError(Exception):
    """Raised when the tape execution encounters an illegal operation."""
    pass


@dataclass
class Cell:
    """A single discrete memory/instruction unit on the 3D Cognitive Tape."""
    type: str           # 'KERNEL', 'IDENTITY', 'MEMORY', 'SCRATCHPAD', etc.
    content: str        # The actual text payload
    readonly: bool      # True for KERNEL cells
    timestamp: float    # When this cell was created/last modified


class CognitiveTape:
    """The 3D spatial memory structure — dict of (x,y,z) → Cell."""

    def __init__(self):
        self.cells: Dict[Coord, Cell] = {}

    def write(self, coord: Coord, cell_type: str, content: str,
              readonly: bool = False):
        """Write a cell. Raises if coordinate is readonly."""
        existing = self.cells.get(coord)
        if existing and existing.readonly:
            raise TapeFaultError(
                f"Cell at {coord} is READ-ONLY (type: {existing.type})")

        self.cells[coord] = Cell(
            type=cell_type,
            content=content,
            readonly=readonly,
            timestamp=time.time(),
        )

    def update(self, coord: Coord, content: str):
        """Update content of an existing cell."""
        cell = self.cells.get(coord)
        if not cell:
            raise TapeFaultError(f"No cell at {coord}")
        if cell.readonly:
            raise TapeFaultError(
                f"Cell at {coord} is READ-ONLY (type: {cell.type})")
        cell.content = content
        cell.timestamp = time.time()

    def read(self, coord: Coord) -> Optional[Cell]:
        """Read a cell. Returns None for empty cells (blank symbol)."""
        return self.cells.get(coord)

    def delete(self, coord: Coord):
        """Delete a cell. Cannot delete readonly or empty cells."""
        cell = self.cells.get(coord)
        if not cell:
            raise TapeFaultError(f"Coordinate {coord} is empty")
        if cell.readonly:
            raise TapeFaultError(
                f"Cell at {coord} is READ-ONLY (type: {cell.type})")
        del self.cells[coord]

    def __len__(self) -> int:
        return len(self.cells)

    def to_dict(self) -> dict:
        """Serialize — JSON keys must be strings."""
        return {
            f"{x},{y},{z}": asdict(cell)
            for (x, y, z), cell in self.cells.items()
        }

    @classmethod
    def from_dict(cls, data: Any) -> "CognitiveTape":
        """Deserialize from persistence."""
        tape = cls()
        if isinstance(data, dict):
            for coord_str, cell_data in data.items():
                parts = coord_str.split(",")
                if len(parts) == 3:
                    try:
                        coord = (int(parts[0]), int(parts[1]), int(parts[2]))
                        tape.cells[coord] = Cell(**cell_data)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Skipping malformed cell {coord_str}: {e}")
        return tape


class TapeMachine:
    """
    Manages the Focus Pointer, Tape I/O, and Persistence.

    Core operations: seek, move, read, write, insert, delete, scan
    """

    def __init__(self, user_id: str = "default", scope: str = "PUBLIC",
                 view_radius: int = 5, persist_dir: str = PERSIST_DIR):
        self.user_id = str(user_id)
        self.scope = scope.upper() if scope else "PUBLIC"
        self.tape = CognitiveTape()
        self.focus_pointer: Coord = (0, 0, 0)
        self.view_radius = view_radius

        # Scope-aware paths
        scope_suffix = f"_{self.scope.lower()}" if self.scope != "PUBLIC" else ""
        self.tape_path = os.path.join(
            persist_dir, self.user_id, f"tape{scope_suffix}.json")
        self.backup_path = os.path.join(
            persist_dir, self.user_id, f"tape{scope_suffix}.bak")

        self._load_or_init()
        logger.info(f"TapeMachine initialized: user={self.user_id}, "
                     f"scope={self.scope}, cells={len(self.tape)}")

    # ─── Persistence ──────────────────────────────────────────────

    def _load_or_init(self):
        """Load tape from disk or initialize empty."""
        if os.path.exists(self.tape_path):
            try:
                with open(self.tape_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "cells" in data:
                    self.tape = CognitiveTape.from_dict(data["cells"])
                    fp = data.get("focus_pointer", [0, 0, 0])
                    self.focus_pointer = tuple(fp)
                else:
                    self.tape = CognitiveTape.from_dict(data)
                return
            except Exception as e:
                logger.error(f"Failed to load tape for {self.user_id}: {e}")
                self._restore_from_backup()
                return
        # No file → empty tape with KERNEL cell
        self.tape.write((0, 2, 0), "KERNEL",
                        "You are NeuroForm — an autonomous AI with a "
                        "living Neo4j memory graph.", readonly=True)
        self.save()

    def save(self):
        """Persist current state to disk."""
        try:
            os.makedirs(os.path.dirname(self.tape_path), exist_ok=True)
            with open(self.tape_path, "w", encoding="utf-8") as f:
                json.dump({
                    "focus_pointer": list(self.focus_pointer),
                    "cells": self.tape.to_dict(),
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save tape: {e}")

    def create_snapshot(self):
        """Create backup to .bak file."""
        try:
            os.makedirs(os.path.dirname(self.backup_path), exist_ok=True)
            if os.path.exists(self.tape_path):
                shutil.copy2(self.tape_path, self.backup_path)
                logger.info(f"Tape snapshot created for {self.user_id}")
        except Exception as e:
            logger.error(f"Failed to snapshot tape: {e}")

    def _restore_from_backup(self):
        """Restore from backup file."""
        if os.path.exists(self.backup_path):
            logger.warning(f"Restoring tape for {self.user_id} from backup")
            try:
                shutil.copy2(self.backup_path, self.tape_path)
                with open(self.tape_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and "cells" in data:
                    self.tape = CognitiveTape.from_dict(data["cells"])
                    self.focus_pointer = tuple(data.get("focus_pointer", [0, 0, 0]))
                else:
                    self.tape = CognitiveTape.from_dict(data)
            except Exception as e:
                logger.error(f"Backup restore failed: {e}")
                self.tape = CognitiveTape()
                self.focus_pointer = (0, 0, 0)
        else:
            logger.warning(f"No backup found for {self.user_id}, starting fresh")
            self.tape = CognitiveTape()
            self.focus_pointer = (0, 0, 0)

    # ─── Navigation ───────────────────────────────────────────────

    def op_seek(self, coord: Coord):
        """Jump to an absolute 3D coordinate."""
        self.focus_pointer = coord

    def op_move(self, direction: str):
        """Move relative along Y or Z axes."""
        x, y, z = self.focus_pointer
        d = direction.upper()
        if d == "UP":
            self.focus_pointer = (x, y + 1, z)
        elif d == "DOWN":
            self.focus_pointer = (x, max(0, y - 1), z)
        elif d == "IN":
            self.focus_pointer = (x, y, z + 1)
        elif d == "OUT":
            self.focus_pointer = (x, y, max(0, z - 1))
        else:
            raise TapeFaultError(
                f"Unknown direction '{direction}'. Use UP, DOWN, IN, OUT.")

    # ─── Data Operations ──────────────────────────────────────────

    def op_read(self) -> str:
        """Read current cell. Empty cells return empty string (blank symbol)."""
        cell = self.tape.read(self.focus_pointer)
        return cell.content if cell else ""

    def op_write(self, content: str):
        """Write/overwrite content at current focus pointer."""
        if self.focus_pointer in self.tape.cells:
            cell = self.tape.cells[self.focus_pointer]
            if cell.type in ("IDENTITY", "ARCHITECTURE", "SKILLS"):
                self.create_snapshot()
            self.tape.update(self.focus_pointer, content)
        else:
            self.tape.write(self.focus_pointer, "MEMORY", content)
        self.save()

    def op_insert(self, cell_type: str, content: str):
        """Insert cell at pointer. Shifts existing cells right if occupied."""
        x, y, z = self.focus_pointer
        if self.focus_pointer in self.tape.cells:
            # Shift cells at this (Y,Z) from X onwards to the right
            coords_to_shift = sorted(
                [c for c in self.tape.cells if c[1] == y and c[2] == z and c[0] >= x],
                key=lambda c: c[0], reverse=True
            )
            for c in coords_to_shift:
                new_coord = (c[0] + 1, c[1], c[2])
                self.tape.cells[new_coord] = self.tape.cells[c]
                del self.tape.cells[c]

        self.tape.write(self.focus_pointer, cell_type, content)
        self.save()

    def op_delete(self):
        """Delete cell at current focus pointer."""
        if self.focus_pointer in self.tape.cells:
            cell = self.tape.cells[self.focus_pointer]
            if cell.type in ("IDENTITY", "ARCHITECTURE", "SKILLS"):
                self.create_snapshot()
            self.tape.delete(self.focus_pointer)
            self.save()
        else:
            raise TapeFaultError(
                f"Delete failed: {self.focus_pointer} is empty")

    # ─── Search ───────────────────────────────────────────────────

    def op_scan(self, query: str) -> bool:
        """
        Find nearest cell containing query, prioritizing current Z-level.
        Moves focus pointer to match. Returns True if found.
        """
        query_lower = query.lower()
        if not self.tape.cells:
            raise TapeFaultError("Scan failed: tape is empty")

        fx, fy, fz = self.focus_pointer

        # Build Z-level index
        z_levels: Dict[int, list] = {}
        for coord, cell in self.tape.cells.items():
            z = coord[2]
            if z not in z_levels:
                z_levels[z] = []
            z_levels[z].append((coord, cell))

        # Search outward from current Z
        max_z_dist = max(abs(z - fz) for z in z_levels) if z_levels else 0
        for z_dist in range(max_z_dist + 1):
            offsets = [0] if z_dist == 0 else [-z_dist, z_dist]
            for z_offset in offsets:
                target_z = fz + z_offset
                if target_z not in z_levels:
                    continue
                candidates = sorted(
                    z_levels[target_z],
                    key=lambda item: abs(item[0][0] - fx) + abs(item[0][1] - fy)
                )
                for coord, cell in candidates:
                    if (query_lower in cell.type.lower() or
                            query_lower in cell.content.lower()):
                        self.focus_pointer = coord
                        return True

        raise TapeFaultError(f"Scan failed: '{query}' not found")

    # ─── View ─────────────────────────────────────────────────────

    def get_view(self) -> str:
        """Generate LLM prompt showing current tape window."""
        x, y, z = self.focus_pointer
        lines = [
            f"--- TAPE MACHINE (Head at X={x}, Y={y}, Z={z}) ---",
            f"--- Level Y={y}, Thread Z={z} ---",
        ]

        start_x = max(0, x - self.view_radius)
        end_x = x + self.view_radius
        has_content = False

        for i in range(start_x, end_x + 1):
            coord = (i, y, z)
            marker = ">> " if coord == self.focus_pointer else "   "
            if coord in self.tape.cells:
                cell = self.tape.cells[coord]
                lock = " [IMMUTABLE]" if cell.readonly else ""
                preview = cell.content[:200] if len(cell.content) <= 200 else cell.content[:197] + "..."
                lines.append(f"{marker}[{i:03d},{y},{z}] {cell.type}{lock}: {preview}")
                has_content = True
            elif coord == self.focus_pointer:
                lines.append(f"{marker}[{i:03d},{y},{z}] --- BLANK ---")

        if not has_content and self.focus_pointer not in self.tape.cells:
            lines.append("   (This region is blank.)")

        lines.append(f"--- Total: {len(self.tape)} cells ---")
        return "\n".join(lines)

    def get_index(self) -> str:
        """Compact index of all populated cells."""
        if not self.tape.cells:
            return "TAPE INDEX: Empty"

        lines = ["TAPE INDEX:"]
        sorted_coords = sorted(self.tape.cells.keys(),
                                key=lambda c: (c[2], c[1], c[0]))
        for coord in sorted_coords:
            cell = self.tape.cells[coord]
            lock = " [LOCKED]" if cell.readonly else ""
            preview = cell.content[:80].replace("\n", " ")
            if len(cell.content) > 80:
                preview += "..."
            marker = " << HERE" if coord == self.focus_pointer else ""
            lines.append(
                f"  [{coord[0]:03d},{coord[1]},{coord[2]}] "
                f"{cell.type}{lock}: {preview}{marker}")

        lines.append(f"\nTotal: {len(self.tape)} cells | Focus: {self.focus_pointer}")
        return "\n".join(lines)

    def snapshot(self) -> dict:
        """Diagnostic snapshot."""
        return {
            "user_id": self.user_id,
            "scope": self.scope,
            "cell_count": len(self.tape),
            "focus_pointer": self.focus_pointer,
            "tape_path": self.tape_path,
        }
