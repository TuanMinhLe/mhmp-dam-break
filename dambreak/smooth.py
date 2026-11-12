"""Smooth periodic simple-wave reference before gradient blow-up.

This separate benchmark measures the numerical scheme's accuracy on a smooth
solution, which cannot be inferred from a dam break containing a shock.
It uses the constant invariant u-2c=-2*c_ref, c=sqrt(g*h).  Consequently
c_t+(3*c-2*c_ref)*c_x=0.  A characteristic initially labelled a is at
x=a+(3*c_initial(a)-2*c_ref)*t.  Its map remains strictly increasing before
``breaking_time``; monotone bisection then recovers the reference at x.

Cell averages use 16-point Gaussian quadrature, not a claim of exact symbolic
integration.  Increasing quadrature order provides a reference-accuracy check.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import math

import numpy as np

from .exact import _edges, _time


@dataclass(frozen=True)
class SmoothWave:
    """Periodic wet simple wave with c(a,0)=c_ref*(1+amplitude*sin(k*(a-origin))).

    ``amplitude`` is dimensionless.  ``reference_depth`` sets c_ref, and is
    not the exact spatial mean of h: that mean is
    reference_depth*(1+amplitude**2/2).
    """

    reference_depth: float = 0.005
    amplitude: float = 0.05
    gravity: float = 9.81
    length: float = 10.0
    origin: float = -5.0

    def __post_init__(self):
        for field in ("reference_depth", "amplitude", "gravity", "length", "origin"):
            value = float(getattr(self, field))
            if not math.isfinite(value):
                raise ValueError(f"{field} must be finite.")
            object.__setattr__(self, field, value)
        if self.reference_depth <= 0 or self.gravity <= 0 or self.length <= 0:
            raise ValueError("Reference depth, gravity and period length must be positive.")
        if not 0 < self.amplitude < 1:
            raise ValueError("Relative amplitude must satisfy 0 < amplitude < 1.")

    @cached_property
    def c_ref(self) -> float:
        return math.sqrt(self.gravity * self.reference_depth)

    @cached_property
    def wave_number(self) -> float:
        return 2 * math.pi / self.length

    @cached_property
    def breaking_time(self) -> float:
        return 1 / (3 * self.c_ref * self.amplitude * self.wave_number)

    def _initial_c(self, label):
        phase = self.wave_number * (label - self.origin)
        return self.c_ref * (1 + self.amplitude * np.sin(phase))

    def _validate_time(self, time: float) -> float:
        time = _time(time)
        if time >= self.breaking_time:
            raise ValueError("The smooth reference is valid only strictly before breaking_time.")
        return time

    def point_values(self, x, time: float) -> np.ndarray:
        """Return values (h,q) with shape (...,2); periodic in x."""
        time = self._validate_time(time)
        x = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(x)):
            raise ValueError("Coordinates must be finite.")
        # Reducing x modulo the period improves trigonometric accuracy for
        # coordinates many periods from the chosen origin.
        x = self.origin + np.remainder(x - self.origin, self.length)
        if time == 0:
            c = self._initial_c(x)
        else:
            speed_min = self.c_ref * (1 - 3 * self.amplitude)
            speed_max = self.c_ref * (1 + 3 * self.amplitude)
            lower = x - speed_max * time
            upper = x - speed_min * time
            for _ in range(80):
                middle = lower + (upper - lower) / 2
                position = middle + (3 * self._initial_c(middle) - 2 * self.c_ref) * time
                below = position < x
                lower = np.where(below, middle, lower)
                upper = np.where(below, upper, middle)
            label = lower + (upper - lower) / 2
            c = self._initial_c(label)
        h = c * c / self.gravity
        u = 2 * (c - self.c_ref)
        return np.stack((h, h * u), axis=-1)

    def cell_averages(self, edges, time: float, quadrature_order: int = 16) -> np.ndarray:
        """Accurate Gaussian reference averages; increase order to check accuracy."""
        edges = _edges(edges)
        time = self._validate_time(time)
        if (isinstance(quadrature_order, bool) or not isinstance(quadrature_order, (int, np.integer))
                or quadrature_order < 2):
            raise ValueError("quadrature_order must be an integer of at least two.")
        nodes, weights = np.polynomial.legendre.leggauss(quadrature_order)
        width = np.diff(edges)
        center = edges[:-1] + width / 2
        x = center[:, None] + width[:, None] / 2 * nodes
        values = self.point_values(x, time)
        return np.einsum("ijv,j->iv", values, weights) / 2

    def summary(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in (
            "reference_depth", "amplitude", "gravity", "length", "origin",
            "c_ref", "breaking_time")} | {
                "mean_depth": self.reference_depth * (1 + self.amplitude**2 / 2),
                "mean_discharge": 2 * self.reference_depth * self.c_ref * self.amplitude**2,
                "invariant_u_minus_2c": -2 * self.c_ref}
