"""Independent Stoker reference for the wet-bed, initially resting dam break.

The PDE solver must never call this module to advance a numerical state.  This
module supplies initial cell averages and an analytical comparison only.

Coordinates, depths and time are in m, m and s; returned conserved variables
are (h, q=hu), with units m and m**2/s.  The dam is at x=0 by default, as in
the team's ``slide.pdf``.  The reference assumes an infinite flat channel.

Derivation: Ketcheson, LeVeque & del Razo, *Riemann Problems and Jupyter
Solutions*, shallow-water chapter:
https://www.clawpack.org/riemann_book/html/Shallow_water.html
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import math

import numpy as np


def _time(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Time must be finite and nonnegative.")
    return value


def _edges(values) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if (result.ndim != 1 or result.size < 2
            or not np.all(np.isfinite(result))
            or not np.all(np.diff(result) > 0)):
        raise ValueError("Cell edges must be a finite, strictly increasing 1D array.")
    return result


@dataclass(frozen=True)
class StokerProblem:
    """Wet Stoker problem, restricted to h_left > h_right > 0 and u_L=u_R=0."""

    h_left: float = 0.005
    h_right: float = 0.001
    gravity: float = 9.81
    dam_position: float = 0.0

    def __post_init__(self):
        for field in ("h_left", "h_right", "gravity", "dam_position"):
            value = float(getattr(self, field))
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite.")
            object.__setattr__(self, field, value)
        if not self.h_left > self.h_right > 0:
            raise ValueError("Stoker requires h_left > h_right > 0; dry beds are excluded.")
        if self.gravity <= 0:
            raise ValueError("gravity must be strictly positive.")

    @cached_property
    def c_left(self) -> float:
        return math.sqrt(self.gravity * self.h_left)

    @cached_property
    def c_right(self) -> float:
        return math.sqrt(self.gravity * self.h_right)

    def matching_residual(self, depth: float) -> float:
        """Left rarefaction velocity minus right shock velocity, in m/s.

        Rationalizing sqrt(h_left)-sqrt(depth) improves accuracy near equal
        depths.  The residual is strictly decreasing between h_right and h_left.
        """
        depth = float(depth)
        if not math.isfinite(depth) or depth <= 0:
            raise ValueError("The matching depth must be finite and positive.")
        left = (2 * math.sqrt(self.gravity) * (self.h_left - depth)
                / (math.sqrt(self.h_left) + math.sqrt(depth)))
        right = ((depth - self.h_right)
                 * math.sqrt(0.5 * self.gravity * (1 / depth + 1 / self.h_right)))
        return left - right

    @cached_property
    def middle_depth(self) -> float:
        lower, upper = self.h_right, self.h_left
        # Terminate on representable endpoints, not an arbitrary dimensional
        # absolute tolerance that would be inappropriate for millimetre depths.
        for _ in range(1024):
            middle = lower + 0.5 * (upper - lower)
            if middle == lower or middle == upper:
                break
            residual = self.matching_residual(middle)
            if residual == 0:
                return middle
            if residual > 0:
                lower = middle
            else:
                upper = middle
        return min((lower, upper), key=lambda h: abs(self.matching_residual(h)))

    @cached_property
    def c_middle(self) -> float:
        return math.sqrt(self.gravity * self.middle_depth)

    @cached_property
    def middle_velocity(self) -> float:
        return (2 * math.sqrt(self.gravity) * (self.h_left - self.middle_depth)
                / (math.sqrt(self.h_left) + math.sqrt(self.middle_depth)))

    @cached_property
    def head_speed(self) -> float:
        return -self.c_left

    @cached_property
    def tail_speed(self) -> float:
        return self.middle_velocity - self.c_middle

    @cached_property
    def shock_speed(self) -> float:
        # Algebraically equal to h_* u_* / (h_*-h_R), without cancellation
        # in a nearly equal-depth Riemann problem.
        return math.sqrt(0.5 * self.gravity * self.middle_depth
                         * (1 + self.middle_depth / self.h_right))

    def wave_positions(self, time: float) -> tuple[float, float, float]:
        time = _time(time)
        return tuple(self.dam_position + time * speed for speed in
                     (self.head_speed, self.tail_speed, self.shock_speed))

    def first_boundary_time(self, x_min: float, x_max: float) -> float:
        """First exact wave arrival at a boundary containing the initial dam.

        This concerns the exact PDE waves, not the support of numerical
        diffusion.  An outflow boundary can affect numerical errors earlier.
        """
        x_min, x_max = float(x_min), float(x_max)
        if (not math.isfinite(x_min) or not math.isfinite(x_max)
                or not x_min < self.dam_position < x_max):
            raise ValueError("Finite domain boundaries must strictly enclose the dam.")
        return min((self.dam_position - x_min) / self.c_left,
                   (x_max - self.dam_position) / self.shock_speed)

    def point_values(self, x, time: float) -> np.ndarray:
        """Return (..., 2) values (h,q); choose the right trace at the shock."""
        time = _time(time)
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)):
            raise ValueError("Coordinates must be finite.")
        result = np.zeros(x.shape + (2,), dtype=float)
        if time == 0:
            result[..., 0] = np.where(x < self.dam_position, self.h_left, self.h_right)
            return result
        xi = (x - self.dam_position) / time
        left = xi <= self.head_speed
        fan = (xi > self.head_speed) & (xi < self.tail_speed)
        middle = (xi >= self.tail_speed) & (xi < self.shock_speed)
        result[..., 0] = self.h_right
        result[..., 0] = np.where(left, self.h_left, result[..., 0])
        result[..., 0] = np.where(middle, self.middle_depth, result[..., 0])
        result[..., 1] = np.where(middle, self.middle_depth * self.middle_velocity, 0)
        # Indexing before polynomial evaluation avoids evaluating the fan
        # formula at huge xi outside its physical interval when t is tiny.
        fan_xi = xi[fan]
        fan_h = (2 * self.c_left - fan_xi) ** 2 / (9 * self.gravity)
        fan_u = 2 * (self.c_left + fan_xi) / 3
        result[..., 0][fan] = fan_h
        result[..., 1][fan] = fan_h * fan_u
        return result

    def cell_averages(self, edges, time: float) -> np.ndarray:
        """Integrate (h,q) on every cell, including cells crossed by a wave.

        Each cell is intersected with all four smooth regions.  Three-point
        Gaussian quadrature is exact for the quadratic h and cubic q within
        the rarefaction (up to floating-point and matching-solve errors).
        A cell containing the shock is *split*; no quadrature crosses a jump.
        """
        edges = _edges(edges)
        time = _time(time)
        widths = np.diff(edges)
        integrals = np.zeros((len(widths), 2))
        if time == 0:
            left_width = np.clip(self.dam_position - edges[:-1], 0, widths)
            integrals[:, 0] = left_width * self.h_left + (widths - left_width) * self.h_right
            return integrals / widths[:, None]

        head, tail, shock = self.wave_positions(time)
        regions = ((-math.inf, head, "left"), (head, tail, "fan"),
                   (tail, shock, "middle"), (shock, math.inf, "right"))
        nodes = np.array([-math.sqrt(3 / 5), 0.0, math.sqrt(3 / 5)])
        weights = np.array([5 / 9, 8 / 9, 5 / 9])
        for low, high, name in regions:
            a = np.maximum(edges[:-1], low)
            b = np.minimum(edges[1:], high)
            selected = b > a
            lengths = b[selected] - a[selected]
            if name == "fan":
                center = a[selected] + lengths / 2
                xi = ((center[:, None] - self.dam_position)
                      + lengths[:, None] / 2 * nodes) / time
                h = (2 * self.c_left - xi) ** 2 / (9 * self.gravity)
                q = h * (2 / 3) * (self.c_left + xi)
                integrals[selected, 0] += lengths / 2 * (h @ weights)
                integrals[selected, 1] += lengths / 2 * (q @ weights)
            elif name == "middle":
                integrals[selected, 0] += lengths * self.middle_depth
                integrals[selected, 1] += lengths * self.middle_depth * self.middle_velocity
            else:
                depth = self.h_left if name == "left" else self.h_right
                integrals[selected, 0] += lengths * depth
        return integrals / widths[:, None]

    def summary(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in (
            "h_left", "h_right", "gravity", "dam_position", "middle_depth",
            "middle_velocity", "c_left", "c_right", "c_middle", "head_speed",
            "tail_speed", "shock_speed")} | {
                "matching_residual_m_per_s": self.matching_residual(self.middle_depth)}
