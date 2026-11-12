#!/usr/bin/env python3
"""Khảo sát kết hợp N, CFL, tỉ số độ sâu và thang độ sâu tuyệt đối.

Không thay đổi bộ giải. Các phép so sánh luôn dùng trung bình ô của nghiệm
Stoker trước khi sóng chính xác đến biên. Chạy ví dụ từ thư mục dự án:
  python parameter_study.py --output outputs/parameter-study

Nhóm factorial: 2 thông lượng × 4 lưới × 3 CFL × 3 tỉ số = 72 lượt.
Nhóm depth_scale: 2 thông lượng × 3 độ sâu × 2 cách chọn thời gian = 12 lượt.
Hai nhóm có các cấu hình trùng nhau, được ghi rõ trong summary.json.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import math
from pathlib import Path

import numpy as np

from dambreak.exact import StokerProblem
from dambreak.experiments import (
    provenance, save_run, stoker_run, write_csv, write_json,
)
from dambreak.solver import FVConfig


HERE = Path(__file__).resolve().parent
METHODS = ("rusanov", "hll")
GRIDS = (100, 200, 400, 800)
CFLS = (.2, .3, .45)
RATIOS = (.05, .2, .5)
DEPTHS = (.0025, .005, .01)
BASE_DEPTH = .005
BASE_TIME = 6.
LENGTH = 10.


def token(value):
    return f"{value:g}".replace(".", "p")


def configurations():
    """Trả về mọi lượt gọi, kể cả cấu hình chuẩn lặp giữa các nhóm."""
    for method in METHODS:
        for n in GRIDS:
            for cfl in CFLS:
                for ratio in RATIOS:
                    name = f"factorial_{method}_N{n}_C{token(cfl)}_r{token(ratio)}"
                    yield name, "factorial", "fixed_physical_time", (
                        StokerProblem(h_left=BASE_DEPTH, h_right=BASE_DEPTH*ratio),
                        FVConfig(cells=n, cfl=cfl, flux=method), [0., 2., 4., 6.])
    for method in METHODS:
        for depth in DEPTHS:
            for schedule in ("fixed_physical_time", "matched_dimensionless_time"):
                end = BASE_TIME if schedule == "fixed_physical_time" else BASE_TIME*math.sqrt(BASE_DEPTH/depth)
                name = f"depth_{method}_H{token(depth)}_{schedule}"
                yield name, "depth_scale", schedule, (
                    StokerProblem(h_left=depth, h_right=.2*depth),
                    FVConfig(cells=400, cfl=.45, flux=method),
                    [0., end/3., 2.*end/3., end])


def independent_checks(result, problem, record, requested_times):
    """Đối chiếu số liệu đầu ra với các đồng nhất thức độc lập và dữ liệu thực."""
    config = result.config
    expected_edges = np.linspace(config.x_min, config.x_max, config.cells+1)
    np.testing.assert_array_equal(result.edges, expected_edges)
    np.testing.assert_array_equal(result.times, requested_times)
    if result.states.shape != (len(requested_times), config.cells, 2):
        raise AssertionError("Kích thước mảng nghiệm không khớp cấu hình.")
    if not np.isfinite(result.states).all() or not np.all(result.states[..., 0] > 0):
        raise AssertionError("Có trạng thái không hữu hạn hoặc không ướt.")
    if not np.isfinite(result.states[..., 1]/result.states[..., 0]).all():
        raise AssertionError("Vận tốc không hữu hạn.")
    history = result.history
    if [row["step"] for row in history] != list(range(len(history))):
        raise AssertionError("Chỉ số bước không liên tục.")
    if len(history)-1 != record["steps"] or history[-1]["time"] != requested_times[-1]:
        raise AssertionError("Số bước hoặc thời điểm cuối sai.")
    if not all(row["dt"] > 0 for row in history[1:]):
        raise AssertionError("Bước thời gian không dương.")
    if record["max_actual_cfl"] > config.cfl*(1+2e-14):
        raise AssertionError("CFL thực tế vượt hệ số đặt.")
    if record["min_depth_all_steps"] <= 0 or record["depth_corrections"] != 0:
        raise AssertionError("Lượt chạy không giữ nền ướt nguyên vẹn.")
    if requested_times[-1] >= problem.first_boundary_time(config.x_min, config.x_max):
        raise AssertionError("Đối chứng vượt thời điểm sóng đến biên.")

    # Tích phân Stoker phải thỏa M=M0 và P=g(hL²-hR²)t/2 trước sóng tới biên.
    expected_mass = ((problem.dam_position-config.x_min)*problem.h_left
                     +(config.x_max-problem.dam_position)*problem.h_right)
    max_ref_mass, max_ref_momentum = 0., 0.
    scales = (problem.h_left, problem.h_left*problem.c_left)
    for time in result.times:
        exact = problem.cell_averages(result.edges, float(time))
        exact_integral = [math.fsum(float(v)*config.dx for v in exact[:, j]) for j in (0, 1)]
        expected_momentum = .5*problem.gravity*(problem.h_left**2-problem.h_right**2)*float(time)
        max_ref_mass = max(max_ref_mass, abs(exact_integral[0]-expected_mass))
        max_ref_momentum = max(max_ref_momentum, abs(exact_integral[1]-expected_momentum))
    if max_ref_mass > 2e-12*LENGTH*scales[0] or max_ref_momentum > 2e-12*LENGTH*scales[1]:
        raise AssertionError("Tích phân đối chứng không thỏa công thức giải tích.")

    exact = problem.cell_averages(result.edges, requested_times[-1])
    deltas = np.abs(result.states[-1]-exact)
    recomputed = {f"e1_{name}": config.dx*math.fsum(float(x) for x in deltas[:, j])/(LENGTH*scales[j])
                  for j, name in enumerate(("h", "q"))}
    for key, value in recomputed.items():
        if not math.isclose(value, record[key], rel_tol=3e-14, abs_tol=2e-16):
            raise AssertionError(f"Sai số {key} không tái tính được từ nghiệm.")
    return dict(case_id=record["case_id"], passed=True,
                reference_mass_identity_max_abs=max_ref_mass,
                reference_momentum_identity_max_abs=max_ref_momentum,
                e1_h_recomputed=recomputed["e1_h"], e1_q_recomputed=recomputed["e1_q"],
                steps=record["steps"], snapshots=len(result.times))


def compare_original(rows, baseline):
    """So các cấu hình giao nhau, không gộp số liệu khác lịch lưu."""
    import csv
    if baseline is None or not (baseline/"convergence.csv").is_file() or not (baseline/"sensitivity.csv").is_file():
        return {"available": False, "reason": "Không có bộ CSV cơ sở để đối chiếu."}
    lookup = {(r["flux"], r["cells"], r["requested_cfl"], r["depth_ratio"]): r
              for r in rows if r["group"] == "factorial"}
    checks = []
    for filename in ("convergence.csv", "sensitivity.csv"):
        with (baseline/filename).open(newline="", encoding="utf-8") as stream:
            original = list(csv.DictReader(stream))
        for old in original:
            if float(old["time"]) != BASE_TIME:
                continue
            if filename == "convergence.csv":
                ratio = .2
            elif old["case"] == "depth_ratio":
                ratio = float(old["value"])
            elif old["case"] == "cfl":
                ratio = .2
            else:
                continue
            key = (old["flux"], int(old["cells"]), float(old["requested_cfl"]), ratio)
            if key not in lookup:
                continue
            new = lookup[key]
            compared = {name: abs(new[name]-float(old[name])) for name in (
                "e1_h", "e1_q", "e2_h", "e2_q", "einf_h", "einf_q", "shock_position_error")}
            for name in compared:
                if not math.isclose(new[name], float(old[name]), rel_tol=3e-14, abs_tol=2e-16):
                    raise AssertionError(f"Không tái lập số liệu cơ sở: {new['case_id']} / {name}")
            checks.append(dict(case_id=new["case_id"], baseline_file=filename,
                               all_compared_values_exactly_equal=all(x == 0. for x in compared.values()),
                               maximum_absolute_difference=max(compared.values())))
    return dict(available=True, comparisons=len(checks),
                unique_new_cases=len({c["case_id"] for c in checks}), checks=checks,
                all_exactly_equal=all(c["all_compared_values_exactly_equal"] for c in checks))


def compare_similarity(saved):
    records = []
    for method in METHODS:
        base_result, base_problem = saved[(method, BASE_DEPTH, "matched_dimensionless_time")]
        base_state = base_result.states[-1]/np.array([BASE_DEPTH, BASE_DEPTH*base_problem.c_left])
        base_reference = base_problem.cell_averages(base_result.edges, float(base_result.times[-1]))/np.array([BASE_DEPTH, BASE_DEPTH*base_problem.c_left])
        for depth in DEPTHS:
            result, problem = saved[(method, depth, "matched_dimensionless_time")]
            state = result.states[-1]/np.array([depth, depth*problem.c_left])
            reference = problem.cell_averages(result.edges, float(result.times[-1]))/np.array([depth, depth*problem.c_left])
            difference = np.max(np.abs(state-base_state), axis=0)
            reference_difference = np.max(np.abs(reference-base_reference), axis=0)
            if max(difference) > 2e-12 or max(reference_difference) > 2e-12:
                raise AssertionError("Các nghiệm cùng thời gian vô thứ nguyên không thỏa quan hệ đồng dạng.")
            records.append(dict(flux=method, h_left=depth, time=float(result.times[-1]),
                dimensionless_time=float(result.times[-1])*problem.c_left/LENGTH,
                baseline_h_left=BASE_DEPTH, baseline_time=BASE_TIME,
                max_scaled_h_difference=float(difference[0]),
                max_scaled_q_difference=float(difference[1]),
                max_scaled_reference_h_difference=float(reference_difference[0]),
                max_scaled_reference_q_difference=float(reference_difference[1]),
                steps=len(result.history)-1, baseline_steps=len(base_result.history)-1))
    return records


def run(output, baseline):
    output = output.resolve()
    protected = [HERE/"results/study"]
    if baseline is not None:
        protected.append(baseline.resolve())
    if any(output == p.resolve() or p.resolve() in output.parents for p in protected):
        raise ValueError("Không được ghi khảo sát mới vào thư mục dữ liệu cơ sở.")
    output.mkdir(parents=True, exist_ok=True)
    rows, profiles, checks, identities = [], [], [], []
    saved = {}
    for index, (case_id, group, schedule, (problem, config, times)) in enumerate(configurations(), 1):
        result, summary = stoker_run(problem, config, times)
        final_error = summary["errors"][-1]
        wave_positions = problem.wave_positions(times[-1])
        row = dict(case_id=case_id, group=group, schedule=schedule,
            h_left=problem.h_left, h_right=problem.h_right,
            depth_ratio=problem.h_right/problem.h_left,
            x_min=config.x_min, x_max=config.x_max, gravity=config.gravity,
            dam_position=problem.dam_position, boundary=config.boundary,
            dimensionless_time=times[-1]*problem.c_left/LENGTH,
            first_boundary_time=summary["first_exact_boundary_arrival"],
            middle_depth=problem.middle_depth, middle_velocity=problem.middle_velocity,
            head_speed=problem.head_speed, tail_speed=problem.tail_speed,
            shock_speed=problem.shock_speed, head_position=wave_positions[0],
            tail_position=wave_positions[1], shock_position_exact=wave_positions[2],
            **final_error, **summary["diagnostics"])
        row["normalized_mass_residual"] = row["max_abs_mass_residual"]/(LENGTH*problem.h_left)
        row["normalized_momentum_residual"] = row["max_abs_momentum_residual"]/(LENGTH*problem.h_left*problem.c_left)
        row["e1_h_depth_jump"] = row["L1_h"]/(LENGTH*(problem.h_left-problem.h_right))
        rows.append(row)
        checks.append(independent_checks(result, problem, row, times))
        summary["case_id"], summary["study_group"], summary["schedule_type"] = case_id, group, schedule
        save_run(output/"runs"/case_id, result, problem, summary)
        identities.append((tuple(asdict(config).items()), tuple(problem.summary().items()), tuple(times)))
        reference = problem.cell_averages(result.edges, times[-1])
        for i, x in enumerate(result.centers):
            state, exact = result.states[-1, i], reference[i]
            profiles.append(dict(case_id=case_id, group=group, schedule=schedule,
                flux=config.flux, cells=config.cells, requested_cfl=config.cfl,
                h_left=problem.h_left, h_right=problem.h_right,
                depth_ratio=problem.h_right/problem.h_left, time=times[-1],
                dimensionless_time=times[-1]*problem.c_left/LENGTH,
                cell=i, x=float(x), x_dimensionless=float(x)/LENGTH,
                h=float(state[0]), q=float(state[1]),
                h_reference_average=float(exact[0]), q_reference_average=float(exact[1]),
                h_scaled=float(state[0]/problem.h_left),
                q_scaled=float(state[1]/(problem.h_left*problem.c_left)),
                h_reference_scaled=float(exact[0]/problem.h_left),
                q_reference_scaled=float(exact[1]/(problem.h_left*problem.c_left))))
        if group == "depth_scale":
            saved[(config.flux, problem.h_left, schedule)] = result, problem
        print(f"[{index:02d}/84] {case_id}: e1_h={row['e1_h']:.8g}, steps={row['steps']}", flush=True)
    similarity = compare_similarity(saved)
    original = compare_original(rows, baseline)
    for group, filename in (("factorial", "factorial.csv"), ("depth_scale", "depth_scale.csv")):
        write_csv(output/filename, [r for r in rows if r["group"] == group])
    write_csv(output/"profiles-final.csv", profiles)
    write_csv(output/"similarity.csv", similarity)
    write_csv(output/"all-runs.csv", rows)
    summary = dict(provenance=provenance(), number_of_pde_runs=len(rows),
        number_of_unique_configurations=len(set(identities)),
        run_family_counts=dict(Counter(r["group"] for r in rows)),
        methods=list(METHODS), grids=list(GRIDS), cfl_values=list(CFLS), depth_ratios=list(RATIOS),
        absolute_depths=list(DEPTHS), domain=[-5., 5.], gravity=9.81, baseline_depth=BASE_DEPTH,
        factorial_output_times=[0., 2., 4., 6.],
        depth_schedule_definition={"fixed_physical_time": "T=6; output=[0,T/3,2T/3,T]",
            "matched_dimensionless_time": "T=6*sqrt(0.005/H); output=[0,T/3,2T/3,T]"},
        dimensionless_variables={"x": "x/10", "t": "t*sqrt(g*H)/10", "h": "h/H", "q": "q/(H*sqrt(g*H))"},
        minimum_depth=min(r["min_depth_all_steps"] for r in rows),
        maximum_cfl=max(r["max_actual_cfl"] for r in rows),
        maximum_mass_residual=max(r["max_abs_mass_residual"] for r in rows),
        maximum_momentum_residual=max(r["max_abs_momentum_residual"] for r in rows),
        maximum_normalized_mass_residual=max(r["normalized_mass_residual"] for r in rows),
        maximum_normalized_momentum_residual=max(r["normalized_momentum_residual"] for r in rows),
        minimum_boundary_time_margin=min(r["first_boundary_time"]-r["time"] for r in rows),
        depth_corrections=0,
        maximum_scaled_h_difference=max(r["max_scaled_h_difference"] for r in similarity),
        maximum_scaled_q_difference=max(r["max_scaled_q_difference"] for r in similarity),
        original_study_overlap=original,
        scope="Nền ướt, đáy phẳng, không ma sát; kiểm chứng số, chưa kiểm định thực nghiệm.",
        caveats=["84 lượt gọi gồm 80 cấu hình duy nhất; các lượt chuẩn lặp có chủ đích giữa nhóm và lịch.",
            "Thay h_R/h_L làm thay đổi bài toán vật lý; không dùng các hàng đó làm dãy hội tụ lưới.",
            "Cùng thời gian vật lý và cùng thời gian vô thứ nguyên là hai phép so sánh khác nhau.",
            "Tăng H làm tốc độ đặc trưng tăng như sqrt(H), thang q tăng như H^(3/2) với r,g cố định.",
            "Đối chứng Stoker chỉ dùng trước khi sóng đến biên; lịch lưu được đổi đồng dạng trong nhóm matched_dimensionless_time.",
            "Số bước phản ánh số lần cập nhật; không phải phép đo thời gian tính thực tế."])
    write_json(output/"summary.json", summary)
    audit = dict(provenance=provenance(), passed=True, cases_checked=len(checks),
        checks=checks, similarity_tolerance=2e-12,
        original_study_overlap=original,
        source_sha256={str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__), HERE/"dambreak/solver.py", HERE/"dambreak/exact.py", HERE/"dambreak/experiments.py"]},
        data_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(output.glob("*.csv"))})
    write_json(output/"audit.json", audit)
    print(f"Hoàn tất {len(rows)} lượt, {len(set(identities))} cấu hình duy nhất: {output}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=HERE/"results/parameter-study")
    parser.add_argument("--baseline", type=Path, default=HERE/"results/study",
                        help="Thư mục chứa convergence.csv và sensitivity.csv của khảo sát cơ sở; chỉ đọc.")
    args = parser.parse_args()
    run(args.output, args.baseline)


if __name__ == "__main__":
    main()
