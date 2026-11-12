"""Đối chiếu nghiệm tham chiếu bằng số chuẩn và các đồng nhất thức độc lập."""

import unittest

import numpy as np

from dambreak.exact import StokerProblem


class StokerReferenceTests(unittest.TestCase):
    def setUp(self):
        self.problem = StokerProblem()

    def test_middle_state_matches_independent_high_precision_values(self):
        p = self.problem
        self.assertAlmostEqual(p.middle_depth, 0.002539357172283335, delta=3e-17)
        self.assertAlmostEqual(p.middle_velocity, 0.1272797183931022, delta=3e-15)
        self.assertAlmostEqual(p.shock_speed, 0.20996340005244554, delta=3e-15)

    def test_rankine_hugoniot_and_admissible_wave_order(self):
        p = self.problem
        hs, us, hr, g = p.middle_depth, p.middle_velocity, p.h_right, p.gravity
        qs = hs * us
        self.assertAlmostEqual(p.shock_speed * (hs - hr), qs, delta=2e-17)
        self.assertAlmostEqual(
            p.shock_speed * qs,
            qs * us + 0.5 * g * (hs * hs - hr * hr),
            delta=2e-17,
        )
        self.assertLess(p.head_speed, p.tail_speed)
        self.assertLess(p.tail_speed, p.shock_speed)
        self.assertLess(p.c_right, p.shock_speed)
        self.assertLess(p.shock_speed, us + p.c_middle)

    def test_initial_cell_cut_by_dam_uses_volume_fraction(self):
        p = StokerProblem(dam_position=0.13)
        edges = np.array([-0.4, 0.0, 0.2, 0.8])
        actual = p.cell_averages(edges, 0.0)
        expected = np.array([[p.h_left, 0.0], [0.65*p.h_left + 0.35*p.h_right, 0.0], [p.h_right, 0.0]])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-18)

    def test_shock_cut_cell_is_weighted_mixture_not_center_value(self):
        p = self.problem
        t = 6.0
        shock = p.shock_speed * t
        left, right = shock - 0.03, shock + 0.07
        actual = p.cell_averages(np.array([left, right]), t)[0]
        expected = 0.3*np.array([p.middle_depth, p.middle_depth*p.middle_velocity]) + 0.7*np.array([p.h_right, 0.0])
        np.testing.assert_allclose(actual, expected, rtol=2e-14, atol=1e-18)
        self.assertGreater(abs(actual[0] - p.point_values((left+right)/2, t)[0]), 1e-4)

    def test_fan_cell_average_against_independent_antiderivative(self):
        p = self.problem
        t = 6.0
        a, b = -1.0, -0.4
        c, g = p.c_left, p.gravity

        def primitive(x):
            return np.array([
                (4*c*c*x - 2*c*x*x/t + x**3/(3*t*t))/(9*g),
                (8*c**3*x - 2*c*x**3/t**2 + x**4/(2*t**3))/(27*g),
            ])

        np.testing.assert_allclose(
            p.cell_averages(np.array([a, b]), t)[0],
            (primitive(b)-primitive(a))/(b-a),
            rtol=2e-14, atol=1e-18,
        )

    def test_integral_mass_and_momentum_balance_on_nonuniform_grid(self):
        p = self.problem
        edges = np.array([-5.0, -3.8, -1.0, -0.2, 0.0, 0.3, 1.1, 2.8, 5.0])
        widths = np.diff(edges)
        for t in (0.0, 0.125, 2.0, 6.0):
            with self.subTest(t=t):
                integral = np.sum(widths[:, None]*p.cell_averages(edges, t), axis=0)
                expected = np.array([
                    5*(p.h_left+p.h_right),
                    0.5*p.gravity*(p.h_left**2-p.h_right**2)*t,
                ])
                np.testing.assert_allclose(integral, expected, rtol=3e-14, atol=3e-17)

    def test_point_values_regions_and_translation_invariance(self):
        p = self.problem
        shifted = StokerProblem(dam_position=5.0)
        x = np.linspace(-4, 4, 31)
        for t in (0.0, 2.0, 6.0):
            with self.subTest(t=t):
                np.testing.assert_allclose(p.point_values(x, t), shifted.point_values(x+5, t), rtol=2e-14, atol=2e-18)
        np.testing.assert_array_equal(p.point_values(np.array([-10., 10.]), 6), np.array([[p.h_left, 0.], [p.h_right, 0.]]))

    def test_boundary_arrival_uses_first_outgoing_wave(self):
        p = self.problem
        self.assertAlmostEqual(p.first_boundary_time(-5.0, 5.0), 5/p.c_left, places=12)
        self.assertAlmostEqual(p.first_boundary_time(-50.0, 1.0), 1/p.shock_speed, places=12)

    def test_invalid_physical_parameters_are_rejected(self):
        for kwargs in (
            {"h_left": 0.0}, {"h_right": 0.0}, {"h_right": -0.01},
            {"h_left": 0.001}, {"h_left": 0.0001},
            {"gravity": 0.0}, {"gravity": -9.81},
            {"h_left": float("nan")}, {"h_right": float("inf")},
            {"dam_position": float("nan")},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((ValueError, TypeError)):
                    StokerProblem(**kwargs)

    def test_invalid_time_and_grid_are_rejected(self):
        p = self.problem
        for t in (-1.0, float("nan"), float("inf")):
            with self.subTest(t=t):
                with self.assertRaises((ValueError, TypeError)):
                    p.point_values(np.array([0.]), t)
        for edges in ([0.], [0., 0.], [1., 0.], [0., float("nan")]):
            with self.subTest(edges=edges):
                with self.assertRaises((ValueError, TypeError)):
                    p.cell_averages(np.array(edges), 0.)


if __name__ == "__main__":
    unittest.main()
