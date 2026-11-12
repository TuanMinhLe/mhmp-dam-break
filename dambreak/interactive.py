"""Chạy PDE và dựng đồ thị tương tác cho phần trình bày trong Jupyter.

Thanh thời gian chọn đúng nghiệm đã lưu; không nội suy nghiệm giữa các mốc.
Đồ thị và bản HTML dùng cùng dữ liệu, không đọc video hoặc kết quả có sẵn.
"""
from dataclasses import dataclass
from html import escape
from pathlib import Path

import numpy as np

from .exact import StokerProblem
from .experiments import errors, conservation_summary
from .solver import FVConfig, SimulationResult, solve
from .style import COLORS, RED, BLACK


METHODS = ("hll", "rusanov")
NAMES = {"hll": "HLL", "rusanov": "Rusanov"}
FONT = "Latin Modern Sans, DejaVu Sans, Arial, sans-serif"
PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
    "toImageButtonOptions": {"filename": "vo_dap_nhom_7", "scale": 2},
}


@dataclass
class InteractiveRun:
    problem: StokerProblem
    results: dict[str, SimulationResult]
    reference: np.ndarray
    error_rows: dict[str, list[dict]]
    reference_valid: np.ndarray | None = None
    flatness: dict[str, list[dict]] | None = None
    stop_info: dict | None = None

    @property
    def first(self):
        return self.results["hll"]


def vi(value, digits=2):
    return f"{value:.{digits}f}".replace(".", ",")


def run_simulation(*, cells=400, h_left_mm=5.0, h_right_mm=1.0,
                   cfl=0.45, final_time=6.0, frames=61):
    """Giải HLL và Rusanov trên cùng lưới và cùng các thời điểm lưu."""
    if not np.isfinite(final_time) or final_time <= 0:
        raise ValueError("Thời gian cuối phải hữu hạn và lớn hơn 0.")
    if isinstance(frames, bool) or not isinstance(frames, (int, np.integer)) or frames < 2:
        raise ValueError("Cần ít nhất hai mốc thời gian, tính cả t = 0.")
    if not (np.isfinite(h_left_mm) and np.isfinite(h_right_mm)
            and h_left_mm > h_right_mm > 0):
        raise ValueError("Cần h_L > h_R > 0; nhập độ sâu theo milimét.")
    problem = StokerProblem(h_left=h_left_mm / 1000,
                            h_right=h_right_mm / 1000)
    configs = {method: FVConfig(cells=cells, cfl=cfl, flux=method)
               for method in METHODS}
    first = configs["hll"]
    boundary_time = problem.first_boundary_time(first.x_min, first.x_max)
    if final_time >= boundary_time:
        raise ValueError(f"Cần T < {vi(boundary_time, 3)} s để đối chiếu trước khi sóng chạm biên.")
    times = np.linspace(0.0, final_time, frames)
    initial = problem.cell_averages(first.edges, 0.0)
    results = {method: solve(initial, config, times)
               for method, config in configs.items()}
    reference = np.stack([problem.cell_averages(first.edges, float(t)) for t in times])
    error_rows = {method: [errors(state, ref, first.dx, problem.h_left, problem.c_left)
                          for state, ref in zip(result.states, reference)]
                  for method, result in results.items()}
    return InteractiveRun(problem, results, reference, error_rows)


def _window(run):
    first = run.first
    if run.stop_info is not None:
        return [float(first.edges[0]), float(first.edges[-1])]
    head, _, shock = run.problem.wave_positions(float(first.times[-1]))
    padding = max(0.2 * (shock - head), 3 * first.config.dx)
    return [max(first.edges[0], head - padding), min(first.edges[-1], shock + padding)]


def make_figure(run, *, frame_duration_ms=100, playback_seconds=None):
    """Đồ thị Plotly với phát/dừng, chọn thời gian, so sánh và thu/phóng."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not np.isfinite(frame_duration_ms) or frame_duration_ms <= 0:
        raise ValueError("Thời gian hiển thị mỗi khung phải lớn hơn 0.")
    first = run.first
    display_indices = np.arange(len(first.times))
    if playback_seconds is not None:
        if not np.isfinite(playback_seconds) or playback_seconds <= 0:
            raise ValueError("Tổng thời gian phát phải hữu hạn và lớn hơn 0.")
        count = min(len(first.times), max(2, round(playback_seconds * 1000 / frame_duration_ms) + 1))
        display_indices = np.rint(np.linspace(0, len(first.times) - 1, count)).astype(int)
        # Plotly queues every selected frame, including the initial state.
        frame_duration_ms = playback_seconds * 1000 / count
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.18)
    spacing = float(np.min(np.diff(first.times)))
    time_digits = max(1, int(np.ceil(-np.log10(spacing) - 1e-9)))
    long_run = run.stop_info is not None

    def title(k):
        heading = "Vỡ đập trên kênh hở" if long_run else "Mô phỏng vỡ đập một chiều"
        if long_run and k == len(first.times) - 1:
            heading += " — Mặt nước gần phẳng" if run.stop_info["reached"] else " — Chưa đạt ngưỡng dừng"
        return (f"<b>{heading}</b>"
                f"<br><span style='font-size:16px'>Nhóm 7 · N = {first.config.cells}"
                f" · CFL = {vi(first.config.cfl)} · t = {vi(first.times[k], max(2, time_digits))} s</span>")

    def annotations(k):
        if long_run:
            depth = " · ".join(f"{NAMES[m]}: {vi(run.flatness[m][k]['depth_span_mm'], 4)} mm"
                               for m in METHODS)
            velocity = " · ".join(f"{NAMES[m]}: {vi(run.flatness[m][k]['mean_velocity'], 4)} m/s"
                                  for m in METHODS)
            before = bool(run.reference_valid[k])
            return [
                dict(text="Độ chênh mặt nước — " + depth, x=1, y=1.035,
                     xref="paper", yref="paper", xanchor="right", yanchor="bottom",
                     showarrow=False, font=dict(size=12, color="#555555")),
                dict(text="Vận tốc trung bình — " + velocity, x=1, y=0.445,
                     xref="paper", yref="paper", xanchor="right", yanchor="bottom",
                     showarrow=False, font=dict(size=12, color="#555555")),
                dict(text="Trước khi sóng chạm biên" if before else "Sau khi sóng chạm biên",
                     x=0, y=1.035, xref="paper", yref="paper", xanchor="left",
                     yanchor="bottom", showarrow=False, font=dict(size=11, color="#666666")),
            ]
        items = []
        for component, y in (("h", 1.035), ("q", 0.445)):
            values = " · ".join(f"{NAMES[m]}: {vi(100 * run.error_rows[m][k]['e1_' + component], 3)} %"
                                for m in METHODS)
            items.append(dict(text="Sai số L¹ chuẩn hóa — " + values,
                              x=1, y=y, xref="paper", yref="paper",
                              xanchor="right", yanchor="bottom", showarrow=False,
                              font=dict(size=12, color="#555555")))
        return items

    def y_values(k):
        values = []
        for component, scale in ((0, 1000), (1, 1e6)):
            ref = run.reference[k, :, component] * scale
            values.append(np.r_[ref, ref[-1]].tolist())
            values.extend((run.results[m].states[k, :, component] * scale).tolist()
                          for m in METHODS)
        return values

    labels = ["Stoker: trung bình ô", "HLL: nghiệm số", "Rusanov: nghiệm số"]
    if long_run:
        labels[0] = f"Stoker: t < {vi(run.stop_info['reference_until_s'])} s"
    colors = [BLACK, COLORS["hll"], COLORS["rusanov"]]
    units = ["mm", "mm²/s"]
    initial_y = y_values(0)
    for row in (1, 2):
        for j in range(3):
            fig.add_trace(go.Scatter(
                x=(first.edges if j == 0 else first.centers).tolist(),
                y=initial_y[3 * (row - 1) + j], name=labels[j],
                mode="lines", legendgroup=str(j), showlegend=(row == 1),
                line=dict(color=colors[j], width=2.2 if j else 1.7,
                          dash="dash" if j == 2 else "solid",
                          shape="hv" if j == 0 else "linear", simplify=False),
                hovertemplate=("x = %{x:.3f} m<br>" + ("h̄" if row == 1 else "q̄")
                               + " = %{y:.3f} " + units[row - 1] + "<extra>%{fullData.name}</extra>"),
            ), row=row, col=1)

    fig.frames = [go.Frame(name=str(k), traces=list(range(6)),
                          data=[go.Scatter(y=y) for y in y_values(k)],
                          layout=go.Layout(title=dict(text=title(k)), annotations=annotations(k)))
                  for k in display_indices]
    instant = dict(mode="immediate", frame=dict(duration=0, redraw=True),
                   transition=dict(duration=0))
    views = [("Vùng có sóng", _window(run)),
             ("Toàn miền", [float(first.edges[0]), float(first.edges[-1])])]
    if long_run:
        center = 0.5 * (first.edges[0] + first.edges[-1])
        half = 0.2 * (first.edges[-1] - first.edges[0])
        views = [("Toàn miền", _window(run)), ("Giữa miền", [center - half, center + half])]
    menus = [dict(type="buttons", direction="left", x=0, y=-0.39,
                  xanchor="left", yanchor="top", showactive=False,
                  buttons=[
                      dict(label="▶ Phát", method="animate", execute=False, args=[None, dict(
                          mode="immediate", fromcurrent=True,
                          frame=dict(duration=frame_duration_ms, redraw=True),
                          transition=dict(duration=0))]),
                      dict(label="Ⅱ Tạm dừng", method="animate", execute=False, args=[[None], instant]),
                      dict(label="↺ Về đầu", method="animate", execute=False, args=[["0"], instant]),
                  ]),
             dict(type="buttons", direction="left", x=0, y=1.16,
                  xanchor="left", yanchor="bottom", showactive=True,
                  buttons=[dict(label=label, method="restyle", args=["visible", visible])
                           for label, visible in [
                               ("Cả hai", [True] * 6),
                               ("HLL", [True, True, False] * 2),
                               ("Rusanov", [True, False, True] * 2)]]),
             dict(type="buttons", direction="left", x=1, y=-0.39,
                  xanchor="right", yanchor="top", showactive=True,
                  buttons=[dict(label=label, method="relayout", args=[{
                      "xaxis.range": limits, "xaxis2.range": limits,
                      "xaxis.autorange": False, "xaxis2.autorange": False}])
                           for label, limits in views])]

    h_values = np.concatenate([run.reference[:, :, 0].ravel()]
                              + [r.states[:, :, 0].ravel() for r in run.results.values()]) * 1000
    q_values = np.concatenate([run.reference[:, :, 1].ravel()]
                              + [r.states[:, :, 1].ravel() for r in run.results.values()]) * 1e6
    def padded(values):
        values = values[np.isfinite(values)]
        span = max(float(np.ptp(values)), 0.05 * float(np.max(np.abs(values))), 1e-12)
        return [float(np.min(values)) - 0.08 * span, float(np.max(values)) + 0.12 * span]

    fig.update_layout(
        template="plotly_white", height=800,
        margin=dict(l=85, r=30, t=160, b=210),
        title=dict(text=title(0), x=0.5, xanchor="center", y=0.98,
                   font=dict(size=23, color=RED)),
        font=dict(family=FONT, size=14, color="#222222"),
        separators=", ", paper_bgcolor="white", plot_bgcolor="white",
        legend=dict(orientation="h", x=1, xanchor="right", y=1.17,
                    yanchor="bottom", font=dict(size=12), groupclick="togglegroup"),
        hovermode="closest", dragmode="zoom", uirevision="keep-view",
        meta=dict(frame_duration_ms=float(frame_duration_ms),
                  playback_seconds_requested=playback_seconds,
                  computed_snapshot_count=len(first.times),
                  displayed_frame_count=len(display_indices),
                  frame_times_s=first.times[display_indices].tolist(),
                  simulation_end_s=float(first.times[-1]),
                  reference_until_s=run.stop_info["reference_until_s"] if long_run else None,
                  near_flat=bool(run.stop_info["reached"]) if long_run else None),
        annotations=annotations(0), updatemenus=menus,
        sliders=[dict(active=0, x=0, len=1, y=-0.08, xanchor="left",
                      pad=dict(t=22, b=0), ticklen=3, font=dict(size=11),
                      currentvalue=dict(prefix="Thời gian: ", suffix=" s", font=dict(size=16, color=RED)),
                      steps=[dict(label=vi(first.times[k], time_digits), method="animate", execute=False,
                                  args=[[str(k)], instant])
                             for k in display_indices])],
    )
    fig.update_xaxes(range=_window(run), showgrid=True, gridcolor="#E8E8E8",
                     zeroline=False, linecolor="#AAAAAA", ticks="outside")
    fig.update_xaxes(title_text="Tọa độ x (m)", row=2, col=1)
    fig.update_yaxes(title_text="Độ sâu h̄ (mm)", range=padded(h_values), row=1, col=1)
    fig.update_yaxes(title_text="Lưu lượng q̄ (mm²/s)", range=padded(q_values), row=2, col=1)
    fig.update_yaxes(gridcolor="#E8E8E8", zeroline=False, linecolor="#AAAAAA")
    return fig


def figure_html(figure):
    """Một tệp HTML chứa cả thư viện đồ thị, dùng được khi không có mạng."""
    # Plotly rejects the pending animation promise when pause/seek cancels it.
    # Handle this expected cancellation locally, preserving unexpected errors.
    # execute=False keeps Plotly's slider binding while we own the promise.
    controls = """
    const plot = document.getElementById('{plot_id}');
    let interruptions = 0;
    plot.on('plotly_animationinterrupted', () => { interruptions += 1; });
    function animate(command) {
        const before = interruptions;
        Plotly.animate(plot, command.args[0], command.args[1]).catch(error => {
            if (error !== undefined || interruptions === before) {
                throw error || new Error('Không thể cập nhật khung hình.');
            }
        });
    }
    plot.on('plotly_buttonclicked', event => {
        if (event.button.method === 'animate') animate(event.button);
    });
    plot.on('plotly_sliderchange', event => {
        // Animation also moves the slider. Only a user's seek may replace
        // the playback queue; reacting to automatic updates stops every frame.
        if (event.interaction === true) animate(event.step);
    });
    """
    return figure.to_html(full_html=True, include_plotlyjs=True,
                          include_mathjax=False, auto_play=False, config=PLOT_CONFIG,
                          post_script=controls)


def display_figure(figure):
    """Nhúng HTML tự chứa trong iframe: không cần tiện ích Jupyter bổ sung."""
    from IPython.display import IFrame, display
    document = figure_html(figure)
    display(IFrame(src="about:blank", width="100%", height=830, extras=[
        'title="Mô phỏng vỡ đập — Nhóm 7"',
        'style="border:0;background:white"',
        f'srcdoc="{escape(document, quote=True)}"',
    ]))


def save_figure(figure, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(figure_html(figure), encoding="utf-8")
    return path


def result_table(run):
    """Bảng tổng kết, ghi rõ sai số tại thời gian cuối và cực trị cả lượt chạy."""
    if run.stop_info is not None:
        info = run.stop_info
        status = "Đạt tiêu chí gần phẳng" if info["reached"] else "Chưa đạt tiêu chí gần phẳng"
        head = ('<tr><th>Thông lượng</th><th>Độ sâu TB (mm)</th>'
                '<th>Chênh mặt nước (mm)</th><th>Lưu lượng TB (mm²/s)</th>'
                '<th>Vận tốc TB (m/s)</th></tr>')
        rows = []
        for method in METHODS:
            m = run.flatness[method][-1]
            rows.append(f'<tr><td>{NAMES[method]}</td><td>{vi(m["mean_depth_mm"], 5)}</td>'
                        f'<td>{vi(m["depth_span_mm"], 6)}</td>'
                        f'<td>{vi(m["mean_discharge"] * 1e6, 4)}</td>'
                        f'<td>{vi(m["mean_velocity"], 6)}</td></tr>')
        return ('<div style="font-family:Arial,sans-serif">'
                f'<p><b>{status} tại t = {vi(run.first.times[-1])} s.</b></p>'
                '<table style="border-collapse:collapse;text-align:right">' + head + ''.join(rows)
                + '</table><p>Mặt nước gần phẳng trong miền [−5, 5] m; nước vẫn chảy qua hai biên.</p>'
                f'<p>Ngưỡng chênh độ sâu: {vi(info["flat_tolerance_mm"], 4)} mm; '
                f'kiểm tra trong ít nhất {vi(info["hold_seconds"], 1)} giây ở các thời điểm lưu. '
                'Đồng thời kiểm tra độ đồng đều của lưu lượng.</p></div>')
    head = ('<tr><th>Thông lượng</th><th>e₁,h (%)</th><th>e₁,q (%)</th>'
            '<th>Số bước</th><th>Độ sâu nhỏ nhất (mm)</th></tr>')
    rows = []
    for method in METHODS:
        diagnostic = conservation_summary(run.results[method])
        err = run.error_rows[method][-1]
        rows.append(f'<tr><td>{NAMES[method]}</td><td>{vi(100 * err["e1_h"], 4)}</td>'
                    f'<td>{vi(100 * err["e1_q"], 4)}</td><td>{diagnostic["steps"]}</td>'
                    f'<td>{vi(1000 * diagnostic["min_depth_all_steps"], 4)}</td></tr>')
    return ('<div style="font-family:Arial,sans-serif">'
            f'<p><b>Sai số tại t = {vi(run.first.times[-1])} s</b></p>'
            '<table style="border-collapse:collapse;text-align:right">' + head + ''.join(rows)
            + '</table><p>Số bước và độ sâu nhỏ nhất được tính trên toàn bộ lượt chạy.</p></div>')
