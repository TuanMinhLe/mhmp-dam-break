"""Bài sóng đơn trơn: kiểm tra tham chiếu và bậc quan sát trước khi tạo sốc."""

import unittest

import numpy as np

from dambreak.smooth import SmoothWave
from dambreak.solver import FVConfig, solve


class SmoothReferenceTests(unittest.TestCase):
    def setUp(self):
        self.wave = SmoothWave()

    def test_values_are_constant_on_characteristics(self):
        w = self.wave
        labels = np.linspace(w.origin, w.origin+w.length, 29)
        c = w.c_ref*(1+w.amplitude*np.sin(2*np.pi*(labels-w.origin)/w.length))
        expected = np.column_stack((c*c/w.gravity, c*c/w.gravity*2*(c-w.c_ref)))
        for t in (0., 5., 20.):
            with self.subTest(t=t):
                positions = labels + (3*c-2*w.c_ref)*t
                np.testing.assert_allclose(w.point_values(positions, t), expected, rtol=3e-13, atol=2e-18)

    def test_constant_riemann_invariant_and_periodicity(self):
        w = self.wave
        x = np.linspace(-5.3, 4.7, 67)
        values = w.point_values(x, 10.)
        invariant = values[:, 1]/values[:, 0]-2*np.sqrt(w.gravity*values[:, 0])
        np.testing.assert_allclose(invariant, -2*w.c_ref, rtol=0, atol=2e-16)
        np.testing.assert_allclose(w.point_values(x+3*w.length, 10.), values, rtol=2e-12, atol=3e-18)

    def test_exact_period_integrals_and_quadrature_accuracy(self):
        w = self.wave
        edges = np.linspace(w.origin, w.origin+w.length, 41)
        expected = w.length*np.array([
            w.reference_depth*(1+w.amplitude**2/2),
            2*w.reference_depth*w.c_ref*w.amplitude**2,
        ])
        for t in (0., 5., 20.):
            with self.subTest(t=t):
                normal = w.cell_averages(edges, t)
                finer = w.cell_averages(edges, t, quadrature_order=32)
                np.testing.assert_allclose(normal, finer, rtol=0, atol=8e-18)
                np.testing.assert_allclose(np.sum(normal, axis=0)*w.length/40, expected, rtol=3e-13, atol=5e-18)

    def test_smooth_reference_satisfies_both_pdes(self):
        w = self.wave
        x = np.linspace(-4.8, 4.8, 37)
        t, eps = 8.0, 1e-4
        temporal = (w.point_values(x, t+eps)-w.point_values(x, t-eps))/(2*eps)

        def flux(values):
            h, q = values[:, 0], values[:, 1]
            return np.column_stack((q, q*q/h+0.5*w.gravity*h*h))

        spatial = (flux(w.point_values(x+eps, t))-flux(w.point_values(x-eps, t)))/(2*eps)
        np.testing.assert_allclose(temporal+spatial, 0., rtol=0, atol=5e-13)

    def test_reference_rejects_times_at_or_after_breaking(self):
        for t in (self.wave.breaking_time, 2*self.wave.breaking_time, -1.):
            with self.subTest(t=t):
                with self.assertRaises(ValueError):
                    self.wave.point_values(np.array([0.]), t)


class SmoothConvergenceTests(unittest.TestCase):
    def test_first_order_refinement_for_each_conserved_component(self):
        w = SmoothWave()
        for method in ("rusanov", "hll"):
            errors = []
            for n in (80, 160, 320):
                config = FVConfig(cells=n, flux=method, boundary="periodic")
                edges = np.linspace(-5., 5., n+1)
                result = solve(w.cell_averages(edges, 0.), config, [6.])
                error = (10/n)*np.sum(np.abs(result.states[-1]-w.cell_averages(edges, 6.)), axis=0)
                errors.append(error)
            orders = np.log2(np.array(errors[:-1])/np.array(errors[1:]))
            for component in (0, 1):
                with self.subTest(method=method, component=component):
                    self.assertGreater(orders[-1, component], 0.8)
                    self.assertLess(orders[-1, component], 1.2)


if __name__ == "__main__":
    unittest.main()
