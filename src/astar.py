from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Sequence, Tuple


from .geometry import Point


Cell = Tuple[int, int]

NEIGHBORS: Sequence[Cell] = (
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
)


def astar_path(env, start: Point, goal: Point) -> List[Point]:
    start_cell = nearest_free_cell(env, env.world_to_grid(start))
    goal_cell = nearest_free_cell(env, env.world_to_grid(goal))
    path_cells = _astar_cells(env, start_cell, goal_cell)
    if path_cells is None:
        raise RuntimeError(f"No path between {start} and {goal}")
    points = [start]
    points.extend(env.grid_to_world(c) for c in path_cells[1:-1])
    points.append(goal)
    return points


def nearest_free_cell(env, cell: Cell) -> Cell:
    gx, gy = cell
    if _is_free(env, gx, gy):
        return cell
    best: Optional[Cell] = None
    best_d = float("inf")
    for radius in range(1, env.cfg.grid_size):
        for y in range(max(0, gy - radius), min(env.cfg.grid_size, gy + radius + 1)):
            for x in range(max(0, gx - radius), min(env.cfg.grid_size, gx + radius + 1)):
                if _is_free(env, x, y):
                    d = abs(x - gx) + abs(y - gy)
                    if d < best_d:
                        best = (x, y)
                        best_d = d
        if best is not None:
            return best
    raise RuntimeError("No free cell in environment")


def _astar_cells(env, start: Cell, goal: Cell) -> Optional[List[Cell]]:
    if start == goal:
        return [start]
    open_heap: List[Tuple[float, int, Cell]] = []
    counter = 0
    heapq.heappush(open_heap, (_heuristic(env, start, goal), counter, start))
    came_from: Dict[Cell, Cell] = {}
    g_score: Dict[Cell, float] = {start: 0.0}
    closed = set()
    while open_heap:
        _, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            return _reconstruct(came_from, current)
        closed.add(current)
        cx, cy = current
        for dx, dy in NEIGHBORS:
            nx, ny = cx + dx, cy + dy
            if not _is_free(env, nx, ny):
                continue
            if dx != 0 and dy != 0:
                if not (_is_free(env, cx + dx, cy) and _is_free(env, cx, cy + dy)):
                    continue
            step = math.hypot(dx, dy) * env.cell_size
            local_cost = 0.5 * (env.base_cost_grid[cy, cx] + env.base_cost_grid[ny, nx])
            tentative = g_score[current] + step * float(local_cost)
            neighbor = (nx, ny)
            if tentative >= g_score.get(neighbor, float("inf")):
                continue
            came_from[neighbor] = current
            g_score[neighbor] = tentative
            counter += 1
            f = tentative + _heuristic(env, neighbor, goal)
            heapq.heappush(open_heap, (f, counter, neighbor))
    return None


def _is_free(env, gx: int, gy: int) -> bool:
    return 0 <= gx < env.cfg.grid_size and 0 <= gy < env.cfg.grid_size and not bool(env.occupancy[gy, gx])


def _heuristic(env, a: Cell, b: Cell) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1]) * env.cell_size


def _reconstruct(came_from: Dict[Cell, Cell], current: Cell) -> List[Cell]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path
