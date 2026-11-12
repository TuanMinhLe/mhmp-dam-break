"""Thí nghiệm tái lập, sai số và xuất dữ liệu; không chứa bước cập nhật PDE."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import platform
import sys
import numpy as np

from .exact import StokerProblem
from .smooth import SmoothWave
from .solver import FVConfig, solve


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        raise ValueError("Không có dữ liệu để xuất CSV.")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def provenance():
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),
                python=sys.version, numpy=np.__version__, platform=platform.platform())


def errors(numerical, reference, dx, depth_scale, velocity_scale):
    """Chuẩn thô có đơn vị; e1/e2/einf đã chuẩn hóa riêng từng thành phần."""
    difference = np.abs(numerical - reference)
    length = dx * len(numerical)
    answer = {}
    for j, name, scale in ((0, "h", depth_scale), (1, "q", depth_scale * velocity_scale)):
        l1 = float(dx * np.sum(difference[:, j]))
        l2 = float(np.sqrt(dx * np.sum(difference[:, j] ** 2)))
        linf = float(np.max(difference[:, j]))
        answer.update({f"L1_{name}": l1, f"L2_{name}": l2, f"Linf_{name}": linf,
                       f"e1_{name}": l1 / (length * scale),
                       f"e2_{name}": l2 / (np.sqrt(length) * scale),
                       f"einf_{name}": linf / scale})
    return answer


def conservation_summary(result):
    history = result.history
    return dict(steps=history[-1]["step"],
        max_actual_cfl=max(row["cfl"] for row in history),
        min_depth_all_steps=min(row["min_depth"] for row in history),
        max_abs_mass_residual=max(abs(row["mass_residual"]) for row in history),
        max_abs_momentum_residual=max(abs(row["momentum_residual"]) for row in history),
        depth_corrections=0)


def shock_estimate(centers, depth, problem, time):
    """Vị trí ngưỡng nửa bước nhảy bằng nội suy tâm ô; chỉ là chẩn đoán."""
    if time == 0:
        return None
    threshold = 0.5 * (problem.middle_depth + problem.h_right)
    crossing = np.flatnonzero((depth[:-1] >= threshold) & (depth[1:] < threshold))
    if not len(crossing):
        return None
    exact = problem.wave_positions(time)[2]
    k = crossing[np.argmin(np.abs(centers[crossing] - exact))]
    fraction = (depth[k] - threshold) / (depth[k] - depth[k + 1])
    return float(centers[k] + fraction * (centers[k + 1] - centers[k]))


def stoker_run(problem, config, times):
    if config.boundary != "outflow":
        raise ValueError("Đối chứng Stoker dùng biên sao chép; không dùng biên tuần hoàn.")
    boundary_time = problem.first_boundary_time(config.x_min, config.x_max)
    if max(times) >= boundary_time:
        raise ValueError(f"Đối chứng Stoker yêu cầu t < {boundary_time:.9g} s trước sóng chạm biên.")
    if config.gravity != problem.gravity:
        raise ValueError("g của bộ giải và đối chứng phải giống nhau.")
    initial = problem.cell_averages(config.edges, 0)
    result = solve(initial, config, times)
    rows = []
    for time, state in zip(result.times, result.states):
        exact = problem.cell_averages(result.edges, time)
        item = dict(flux=config.flux, cells=config.cells, time=float(time),
                    dx=config.dx, requested_cfl=config.cfl)
        item.update(errors(state, exact, config.dx, problem.h_left, problem.c_left))
        shock = shock_estimate(result.centers, state[:, 0], problem, time)
        item["shock_position"] = shock
        item["exact_shock_position"] = problem.wave_positions(time)[2]
        item["shock_position_error"] = None if shock is None else shock - item["exact_shock_position"]
        rows.append(item)
    summary = dict(config=asdict(config), problem=problem.summary(),
        first_exact_boundary_arrival=boundary_time, diagnostics=conservation_summary(result),
        errors=rows, provenance=provenance())
    return result, summary


def save_run(directory, result, problem, summary):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    profiles = []
    for time, state in zip(result.times, result.states):
        reference = problem.cell_averages(result.edges, float(time))
        for i, x in enumerate(result.centers):
            profiles.append(dict(time=float(time), cell=i, x_left=result.edges[i],
                x_right=result.edges[i + 1], x=x, h=state[i, 0], q=state[i, 1],
                u_from_averages=state[i, 1] / state[i, 0],
                h_reference_average=reference[i, 0], q_reference_average=reference[i, 1],
                u_reference_from_averages=reference[i, 1] / reference[i, 0]))
    write_csv(directory / "profiles.csv", profiles)
    write_csv(directory / "history.csv", result.history)
    write_json(directory / "summary.json", summary)


def run_study(directory, make_plots=True):
    """Ma trận T1–T5 của đề xuất: số liệu thật, không thay bằng bậc lý thuyết."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    grids = [100, 200, 400, 800, 1600]
    methods = ["rusanov", "hll"]
    problem, wave = StokerProblem(), SmoothWave()
    convergence, smooth, sensitivity, domain_checks = [], [], [], []
    diagnostic_records, comparison, base = [], {}, {}
    for method in methods:
        previous = {}
        for n in grids:
            config = FVConfig(cells=n, flux=method)
            result, summary = stoker_run(problem, config, [0, 2, 4, 6])
            save_run(directory / "runs" / f"stoker_{method}_{n}", result, problem, summary)
            diagnostic_records.append(dict(case=f"stoker_{method}_{n}", **summary["diagnostics"]))
            for row in summary["errors"][1:]:
                row = row.copy()
                for name in ("h", "q"):
                    old = previous.get((row["time"], name))
                    row[f"p_{name}"] = None if old is None else float(np.log2(old / row[f"e1_{name}"]))
                    previous[(row["time"], name)] = row[f"e1_{name}"]
                convergence.append(row)
            if n == 400:
                comparison[method] = result
                base[method] = result
            print(f"Stoker: {method:7s}, N={n:4d}, e1_h(t=6)={summary['errors'][-1]['e1_h']:.5g}", flush=True)

        previous = None
        for n in grids:
            config = FVConfig(cells=n, flux=method, boundary="periodic")
            result = solve(wave.cell_averages(config.edges, 0), config, [0, 5])
            row = dict(flux=method, cells=n, dx=config.dx, time=5.0, requested_cfl=config.cfl)
            row.update(errors(result.states[-1], wave.cell_averages(config.edges, 5),
                              config.dx, wave.reference_depth, wave.c_ref))
            for name in ("h", "q"):
                row[f"p_{name}"] = None if previous is None else float(np.log2(previous[f"e1_{name}"] / row[f"e1_{name}"]))
            smooth.append(row)
            previous = row
            diagnostics = conservation_summary(result)
            diagnostic_records.append(dict(case=f"smooth_{method}_{n}", **diagnostics))
            save_run(directory / "runs" / f"smooth_{method}_{n}", result, wave,
                     dict(config=asdict(config), problem=wave.summary(), errors=[row],
                          diagnostics=diagnostics, provenance=provenance()))
            print(f"Sóng trơn: {method:7s}, N={n:4d}, e1_h={row['e1_h']:.5g}", flush=True)

        cases = [("cfl", c, problem, FVConfig(cells=400, flux=method, cfl=c)) for c in [.2, .3, .45]]
        cases += [("depth_ratio", r, StokerProblem(h_right=.005*r), FVConfig(cells=400, flux=method))
                  for r in [.01, .05, .1, .2, .5, .9]]
        cases += [("off_grid", 401, problem, FVConfig(cells=401, flux=method))]
        for kind, value, case_problem, config in cases:
            result, summary = stoker_run(case_problem, config, [0, 2, 4, 6])
            name = f"{kind}_{method}_{value:g}"
            save_run(directory / "runs" / name, result, case_problem, summary)
            diagnostic_records.append(dict(case=name, **summary["diagnostics"]))
            sensitivity.append(dict(case=kind, value=value, **summary["errors"][-1],
                                    **summary["diagnostics"]))

        # Cùng dx và lưới lồng nhau: so miền [-5,5] với phần giữa của [-10,10].
        config = FVConfig(x_min=-10, x_max=10, cells=800, flux=method)
        expanded, summary = stoker_run(problem, config, [0, 2, 4, 6])
        save_run(directory / "runs" / f"domain_{method}", expanded, problem, summary)
        diagnostic_records.append(dict(case=f"domain_{method}", **summary["diagnostics"]))
        delta = np.abs(expanded.states[-1, 200:600] - base[method].states[-1])
        domain_checks.append(dict(flux=method, compared_time=6.0, dx=config.dx,
            max_h_difference=float(delta[:, 0].max()), max_q_difference=float(delta[:, 1].max())))

    write_csv(directory / "convergence.csv", convergence)
    write_csv(directory / "smooth_convergence.csv", smooth)
    write_csv(directory / "sensitivity.csv", sensitivity)
    write_csv(directory / "diagnostics.csv", diagnostic_records)
    write_csv(directory / "domain_checks.csv", domain_checks)
    summary = dict(provenance=provenance(), grids=grids, methods=methods,
        stoker_problem=problem.summary(), smooth_problem=wave.summary(),
        cfl=.45, stoker_times=[0, 2, 4, 6], smooth_time=5,
        number_of_pde_runs=len(diagnostic_records),
        minimum_depth=min(r["min_depth_all_steps"] for r in diagnostic_records),
        maximum_cfl=max(r["max_actual_cfl"] for r in diagnostic_records),
        maximum_mass_residual=max(r["max_abs_mass_residual"] for r in diagnostic_records),
        maximum_momentum_residual=max(r["max_abs_momentum_residual"] for r in diagnostic_records),
        depth_corrections=0, domain_checks=domain_checks,
        scope="Nền ướt, đáy phẳng, không ma sát; kiểm chứng số, chưa kiểm định thực nghiệm.")
    write_json(directory / "summary.json", summary)
    if make_plots:
        from .plotting import plot_study
        plot_study(directory, comparison, problem, convergence, smooth)
    return summary
