"""Kiểm chứng bộ giải PDE bằng bất biến vật lý và bài có nghiệm biết trước."""

import unittest

import numpy as np

from dambreak.exact import StokerProblem
from dambreak.solver import FVConfig, numerical_flux, physical_flux, solve


METHODS = ("rusanov", "hll")


class FluxTests(unittest.TestCase):
    def test_consistency_with_physical_flux(self):
        h = np.array([0.001, 0.005, 0.2, 1.0, 2.0])
        u = np.array([-5.0, 0.0, 0.3, -0.7, 8.0])
        state = np.column_stack((h, h*u))
        expected = np.column_stack((h*u, h*u*u + 0.5*9.81*h*h))
        np.testing.assert_allclose(physical_flux(state), expected, rtol=3e-15, atol=1e-18)
        for method in METHODS:
            with self.subTest(method=method):
                flux, speed = numerical_flux(state, state, method=method)
                np.testing.assert_allclose(flux, expected, rtol=3e-15, atol=1e-18)
                np.testing.assert_allclose(speed, np.abs(u)+np.sqrt(9.81*h), rtol=3e-15)

    def test_hll_selects_upwind_flux_for_supercritical_flow(self):
        for direction in (-1., 1.):
            with self.subTest(direction=direction):
                left = np.array([[0.1, 0.1*direction*4.]])
                right = np.array([[0.2, 0.2*direction*5.]])
                expected_state = left if direction > 0 else right
                actual, _ = numerical_flux(left, right, method="hll")
                np.testing.assert_allclose(actual, physical_flux(expected_state), rtol=3e-15)

    def test_reflection_symmetry_of_flux(self):
        left = np.array([[0.005, 0.001], [0.02, -0.003]])
        right = np.array([[0.001, -0.0002], [0.03, 0.002]])
        reflection = np.array([1., -1.])
        for method in METHODS:
            with self.subTest(method=method):
                forward, speed1 = numerical_flux(left, right, method=method)
                reflected, speed2 = numerical_flux(right*reflection, left*reflection, method=method)
                np.testing.assert_allclose(reflected, forward*np.array([-1., 1.]), rtol=3e-15, atol=1e-18)
                np.testing.assert_allclose(speed1, speed2, rtol=3e-15)


class EvolutionTests(unittest.TestCase):
    def test_uniform_states_are_preserved_for_both_boundaries(self):
        for method in METHODS:
            for boundary in ("outflow", "periodic"):
                for velocity in (0.0, -0.17, 0.21):
                    with self.subTest(method=method, boundary=boundary, velocity=velocity):
                        config = FVConfig(cells=37, flux=method, boundary=boundary)
                        initial = np.tile([0.005, 0.005*velocity], (37, 1))
                        result = solve(initial, config, [0.0, 0.17, 2.0])
                        np.testing.assert_array_equal(result.states, np.broadcast_to(initial, result.states.shape))

    def test_output_times_are_exact_and_input_and_snapshots_independent(self):
        config = FVConfig(cells=51)
        edges = np.linspace(config.x_min, config.x_max, config.cells+1)
        initial = StokerProblem(dam_position=0.013).cell_averages(edges, 0.0)
        saved = initial.copy()
        times = [0.0, 0.013, 0.17, 0.203]
        result = solve(initial, config, times)
        np.testing.assert_array_equal(result.times, times)
        self.assertEqual(result.history[-1]["time"], times[-1])
        np.testing.assert_array_equal(initial, saved)
        np.testing.assert_array_equal(result.states[0], saved)
        self.assertFalse(np.shares_memory(result.states, initial))
        self.assertGreater(np.max(np.abs(result.states[-1]-result.states[0])), 0)

    def test_first_requested_time_need_not_be_zero(self):
        config = FVConfig(cells=40)
        edges = np.linspace(config.x_min, config.x_max, 41)
        initial = StokerProblem().cell_averages(edges, 0.)
        first = solve(initial, config, [0.123])
        second = solve(initial, config, [0., 0.123])
        np.testing.assert_array_equal(first.states[0], second.states[-1])
        self.assertEqual(first.history[0]["time"], 0.)

    def test_boundary_flux_ledger_reconstructs_total_change(self):
        for method in METHODS:
            with self.subTest(method=method):
                config = FVConfig(cells=63, x_min=-1, x_max=1, flux=method)
                x = np.linspace(-1+1/63, 1-1/63, 63)
                h = 0.02 + 0.003*x
                initial = np.column_stack((h, h*(0.3+0.08*x)))
                result = solve(initial, config, [0., 0.73])
                dx = 2/63
                net = np.zeros(2)
                for row in result.history[1:]:
                    net += row["dt"]*np.array([
                        row["flux_left_h"]-row["flux_right_h"],
                        row["flux_left_q"]-row["flux_right_q"],
                    ])
                    observed = np.array([row["mass"], row["momentum"]])-dx*np.sum(initial, axis=0)
                    np.testing.assert_allclose(observed, net, rtol=0, atol=3e-16)
                    self.assertGreater(row["min_depth"], 0.)
                    self.assertLessEqual(row["cfl"], config.cfl*(1+2e-14))
                np.testing.assert_allclose(dx*np.sum(result.states[-1]-initial, axis=0), net, rtol=0, atol=3e-16)
                self.assertGreater(abs(net[0]), 1e-6)
                self.assertGreater(abs(net[1]), 1e-6)

    def test_periodic_mass_and_momentum_are_conserved(self):
        rng = np.random.default_rng(20261029)
        h = rng.uniform(0.001, 0.02, 79)
        initial = np.column_stack((h, h*rng.uniform(-0.5, 0.5, 79)))
        for method in METHODS:
            with self.subTest(method=method):
                config = FVConfig(cells=79, boundary="periodic", flux=method)
                result = solve(initial, config, [0., 0.8])
                np.testing.assert_allclose(np.sum(result.states[-1], axis=0), np.sum(initial, axis=0), rtol=3e-14, atol=2e-16)
                for row in result.history:
                    self.assertGreater(row["min_depth"], 0.)
                    self.assertAlmostEqual(row["cumulative_mass_inflow"], 0., delta=2e-17)
                    self.assertAlmostEqual(row["cumulative_momentum_inflow"], 0., delta=2e-17)

    def test_stoker_momentum_grows_by_boundary_pressure_difference(self):
        p = StokerProblem()
        for method in METHODS:
            with self.subTest(method=method):
                config = FVConfig(cells=160, flux=method)
                edges = np.linspace(-5., 5., 161)
                initial = p.cell_averages(edges, 0.)
                result = solve(initial, config, [0., 2.])
                total = (10/160)*np.sum(result.states[-1], axis=0)
                expected = [5*(p.h_left+p.h_right), p.gravity*(p.h_left**2-p.h_right**2)]
                np.testing.assert_allclose(total, expected, rtol=5e-13, atol=2e-16)
                self.assertGreater(total[1], 0.)

    def test_positive_depth_in_diverging_flow_without_mass_injection(self):
        # Bài giãn mạnh từng khiến bộ giải Roe cơ bản sinh h trung gian âm.
        for method in METHODS:
            with self.subTest(method=method):
                config = FVConfig(cells=100, x_min=-5, x_max=5, gravity=1., flux=method, cfl=0.5)
                initial = np.ones((100, 2))
                initial[:50, 1] = -1.5
                initial[50:, 1] = 1.5
                result = solve(initial, config, [0., 0.7])
                self.assertGreater(np.min(result.states[..., 0]), 0.)
                self.assertLess(np.min(result.states[-1, :, 0]), 0.5)
                self.assertLess(max(abs(r["mass_residual"]) for r in result.history), 2e-13)
                self.assertLess(max(abs(r["momentum_residual"]) for r in result.history), 2e-13)

    def test_one_step_positivity_for_varied_wet_states_at_cfl_limit(self):
        rng = np.random.default_rng(1739)
        for method in METHODS:
            for sample in range(12):
                with self.subTest(method=method, sample=sample):
                    h = 10**rng.uniform(-6., 0., 48)
                    velocity = rng.uniform(-5., 5., 48)
                    initial = np.column_stack((h, h*velocity))
                    config = FVConfig(cells=48, x_min=0., x_max=1., flux=method,
                                      boundary="periodic", cfl=0.5)
                    dt = 0.5/48/np.max(np.abs(velocity)+np.sqrt(9.81*h))
                    result = solve(initial, config, [dt])
                    self.assertGreater(np.min(result.states[..., 0]), 0.)
                    np.testing.assert_allclose(np.sum(result.states[-1], axis=0),
                                               np.sum(initial, axis=0), rtol=3e-14, atol=3e-15)

    def test_dam_break_errors_decrease_under_grid_refinement(self):
        p = StokerProblem()
        for method in METHODS:
            errors = []
            for n in (100, 200, 400):
                config = FVConfig(cells=n, flux=method)
                edges = np.linspace(-5, 5, n+1)
                result = solve(p.cell_averages(edges, 0.), config, [6.])
                errors.append((10/n)*np.sum(np.abs(result.states[-1]-p.cell_averages(edges, 6.)), axis=0))
            for variable in (0, 1):
                with self.subTest(method=method, variable=variable):
                    self.assertGreater(errors[0][variable], errors[1][variable])
                    self.assertGreater(errors[1][variable], errors[2][variable])
                    self.assertLess(errors[2][variable], 0.8*errors[0][variable])


class InputValidationTests(unittest.TestCase):
    def test_invalid_configuration_is_rejected(self):
        for kwargs in (
            {"cells": 0}, {"cells": -2}, {"cells": 2.5},
            {"x_min": 1., "x_max": 1.}, {"x_min": 2., "x_max": 1.},
            {"x_min": float("nan")}, {"gravity": 0.},
            {"cfl": 0.}, {"cfl": -0.1}, {"cfl": 0.51}, {"cfl": float("nan")},
            {"flux": "unknown"}, {"boundary": "unknown"}, {"max_steps": 0},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((ValueError, TypeError)):
                    FVConfig(**kwargs)

    def test_invalid_states_are_rejected(self):
        config = FVConfig(cells=10)
        good = np.tile([0.005, 0.], (10, 1))
        bad_states = [np.ones((9, 2)), np.ones((10, 3))]
        for component, value in ((0, 0.), (0, -0.001), (0, float("nan")), (1, float("inf"))):
            bad = good.copy()
            bad[4, component] = value
            bad_states.append(bad)
        for index, state in enumerate(bad_states):
            with self.subTest(index=index):
                with self.assertRaises((ValueError, TypeError)):
                    solve(state, config, [0.1])

    def test_invalid_output_times_are_rejected(self):
        config = FVConfig(cells=10)
        initial = np.tile([0.005, 0.], (10, 1))
        for times in ([], [-0.1], [0.2, 0.1], [0.1, 0.1], [float("nan")], [float("inf")]):
            with self.subTest(times=times):
                with self.assertRaises((ValueError, TypeError)):
                    solve(initial, config, times)

    def test_step_limit_does_not_return_partial_success(self):
        config = FVConfig(cells=100, max_steps=1)
        edges = np.linspace(-5., 5., 101)
        with self.assertRaises(RuntimeError):
            solve(StokerProblem().cell_averages(edges, 0.), config, [100.])


if __name__ == "__main__":
    unittest.main()
