"""Hình khoa học và hoạt hình, tách khỏi bộ giải và được nhập khi cần.

Đường đối chứng trong hình biên dạng cũng là trung bình ô. Vận tốc hiển thị
là q_bar/h_bar, không gọi là trung bình của vận tốc điểm. Hoạt hình chạy
lại PDE với lịch xuất dày hơn và lưu đầy đủ dữ liệu của chính lần chạy đó.
"""

from pathlib import Path
from subprocess import CalledProcessError

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.ticker import MaxNLocator, NullFormatter
import numpy as np

from .experiments import save_run, stoker_run
from .style import apply_style, apply_figure_fonts, COLORS, RED, BLACK, GRAY


NAMES = {"hll": "HLL", "rusanov": "Rusanov"}


def _style():
    apply_style()


def _save(fig, directory, name):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    apply_figure_fonts(fig)
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight", pad_inches=.16)
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight", pad_inches=.16)
    plt.close(fig)


def _variables(state):
    return (state[:, 0] * 1e3, state[:, 1] / state[:, 0], state[:, 1] * 1e6)


LABELS = (r"Độ sâu $\overline{h}$ (mm)",
          r"Vận tốc $\overline{q}/\overline{h}$ (m/s)",
          r"Lưu lượng $\overline{q}$ (mm²/s)")


def _window(problem, result):
    time = float(result.times[-1])
    if time <= 0:
        return result.edges[0], result.edges[-1]
    head, _, shock = problem.wave_positions(time)
    width = shock - head
    return (max(result.edges[0], head - .2 * width),
            min(result.edges[-1], shock + .2 * width))


def _profile_axis(ax, result, reference, index, method, reference_label=True):
    values = _variables(result.states[-1])[index]
    exact_values = _variables(reference)[index]
    if reference_label:
        ax.stairs(exact_values, result.edges, color=BLACK, linewidth=1.45, baseline=None,
                  label="Stoker: trung bình ô", zorder=3)
    ax.plot(result.centers, values, color=COLORS[method], linewidth=1.65,
            linestyle="--" if method == "rusanov" else "-",
            label=f"{NAMES[method]}: nghiệm số", zorder=4)
    ax.set_ylabel(LABELS[index])
    ax.grid(True, alpha=.8)
    ax.yaxis.set_major_locator(MaxNLocator(5))


def plot_run(directory, result, problem):
    """Lưu profile.pdf/png cho lần chạy đã có, không chạy lại bộ giải."""
    _style()
    time = float(result.times[-1])
    reference = problem.cell_averages(result.edges, time)
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.8), sharex=True,
                             layout="constrained")
    for index, ax in enumerate(axes):
        _profile_axis(ax, result, reference, index, result.config.flux)
        ax.set_xlim(*_window(problem, result))
    axes[0].legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Tọa độ x (m)")
    fig.suptitle(f"Vỡ đập trên nền ướt — {NAMES[result.config.flux]}, "
                 f"N = {result.config.cells}, t = {time:g} s\n"
                 f"Miền tính [{result.edges[0]:g}, {result.edges[-1]:g}] m; "
                 "hình tập trung vào vùng có sóng", fontsize=13, fontweight="bold", color=RED)
    _save(fig, directory, "profile")


def _comparison(directory, comparison, problem):
    first = next(iter(comparison.values()))
    time = float(first.times[-1])
    reference = problem.cell_averages(first.edges, time)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), layout="constrained")
    flat = axes.flat
    for index, ax in enumerate(list(flat)[:3]):
        for k, (method, result) in enumerate(comparison.items()):
            _profile_axis(ax, result, reference, index, method, reference_label=(k == 0))
        ax.set_xlim(*_window(problem, first))
        ax.set_xlabel("Tọa độ x (m)")
    zoom = axes[1, 1]
    for k, (method, result) in enumerate(comparison.items()):
        _profile_axis(zoom, result, reference, 0, method, reference_label=(k == 0))
    shock = problem.wave_positions(time)[2]
    half_width = max(12 * first.config.dx, .12 * (problem.wave_positions(time)[2]
                                               - problem.wave_positions(time)[0]))
    zoom.set_xlim(max(first.edges[0], shock - half_width),
                  min(first.edges[-1], shock + half_width))
    zoom.set_ylim(.85 * problem.h_right * 1e3, 1.09 * problem.middle_depth * 1e3)
    zoom.axvline(shock, color=GRAY, linewidth=.9, linestyle=":", label="Vị trí sốc chính xác")
    zoom.set_title("Phóng to sóng xung kích")
    zoom.set_xlabel("Tọa độ x (m)")
    zoom.legend(loc="upper right", fontsize=8)
    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"So sánh hai thông lượng — N = {first.config.cells}, t = {time:g} s\n"
                 "Đối chứng theo trung bình ô; đường nối tâm ô biểu diễn nghiệm số",
                 fontsize=13, fontweight="bold", color=RED)
    _save(fig, directory, "profile_comparison")


def _convergence(directory, rows, smooth=False):
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.1), layout="constrained")
    times = sorted({float(row["time"]) for row in rows})
    styles = {time: style for time, style in zip(times, ("-", "--", ":", "-."))}
    for component, ax in zip(("h", "q"), axes):
        for method in ("rusanov", "hll"):
            for time in times:
                selected = sorted((row for row in rows if row["flux"] == method
                                   and float(row["time"]) == time), key=lambda row: row["cells"])
                if not selected:
                    continue
                x = np.array([row["cells"] for row in selected])
                y = np.array([row[f"e1_{component}"] for row in selected])
                line_style = ("--" if method == "rusanov" else "-") if smooth else styles[time]
                ax.loglog(x, y, color=COLORS[method], linestyle=line_style,
                          marker="o" if method == "hll" else "s", markersize=4.5,
                          linewidth=1.65,
                          label=NAMES[method] if smooth else f"{NAMES[method]}, t = {time:g} s")
        if smooth:
            cells = np.array(sorted({row["cells"] for row in rows}))
            largest_coarse_error = max(row[f"e1_{component}"] for row in rows
                                      if row["cells"] == cells[0])
            guide = 1.65 * largest_coarse_error * cells[0] / cells
            ax.loglog(cells, guide, color=GRAY, linewidth=1.1, linestyle="--",
                      label=r"Đường chuẩn $N^{-1}$ (đã dịch)")
        ax.set_xlabel("Số ô N")
        ax.set_ylabel(rf"Sai số $e_{{1,{component}}}$ (không thứ nguyên)")
        cells = sorted({row["cells"] for row in rows})
        ax.set_xticks(cells, labels=[f"{n:g}" for n in cells])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, which="major", alpha=.8)
        ax.grid(True, which="minor", alpha=.25)
        ax.legend(loc="upper right", fontsize=8, ncol=1 if smooth else 2)
    if smooth:
        title = "Hội tụ trên sóng đơn trơn, tuần hoàn — t = 5 s"
        subtitle = "Đối chứng trước thời điểm mất trơn; đường chuẩn chỉ biểu diễn độ dốc bậc một"
        name = "smooth_convergence"
    else:
        title = "Sai số theo lưới của bài toán Stoker có sốc"
        subtitle = "CFL = 0,45; so sánh các biến bảo toàn với trung bình ô chính xác"
        name = "convergence"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=12.5, fontweight="bold", color=RED)
    _save(fig, directory, name)


def _diagnostics(directory, comparison, problem):
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8), layout="constrained")
    for method, result in comparison.items():
        time = np.array([row["time"] for row in result.history])
        style = dict(color=COLORS[method], linewidth=1.5,
                     linestyle="--" if method == "rusanov" else "-", label=NAMES[method])
        axes[0, 0].plot(time, [row["mass_residual"] for row in result.history], **style)
        axes[0, 1].plot(time, [row["momentum_residual"] for row in result.history], **style)
        axes[1, 0].plot(time, [row["momentum"] for row in result.history], **style)
        axes[1, 1].plot(time, [row["min_depth"] * 1e3 for row in result.history], **style)
    last_time = max(float(result.times[-1]) for result in comparison.values())
    time = np.linspace(0, last_time, 100)
    momentum = .5 * problem.gravity * (problem.h_left**2 - problem.h_right**2) * time
    axes[1, 0].plot(time, momentum, color=BLACK, linewidth=1.2, linestyle=":",
                    label="Đối chứng cân bằng áp lực")
    axes[1, 1].axhline(problem.h_right * 1e3, color=BLACK, linewidth=1.1,
                       linestyle=":", label=r"Độ sâu hạ lưu $h_R$")
    labels = (("Số dư cân bằng lượng nước", r"$R_h$ (m²)"),
              ("Số dư cân bằng động lượng", r"$R_q$ (m³/s)"),
              ("Động lượng tăng do áp lực ở hai biên", r"$\int q\,dx$ (m³/s)"),
              ("Độ sâu nhỏ nhất trong từng bước", r"$\min_i\overline{h}_i$ (mm)"))
    for ax, (title, label) in zip(axes.flat, labels):
        ax.set_title(title)
        ax.set_ylabel(label)
        ax.set_xlabel("Thời gian t (s)")
        ax.grid(True, alpha=.8)
        ax.legend(loc="best", fontsize=8)
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    cells = next(iter(comparison.values())).config.cells
    fig.suptitle(f"Kiểm tra bảo toàn và độ sâu — N = {cells}\n"
                 "Số dư dùng thông lượng biên thực tế; không cắt hoặc sửa độ sâu",
                 fontsize=13, fontweight="bold", color=RED)
    _save(fig, directory, "diagnostics")


def plot_study(directory, comparison, problem, convergence, smooth):
    """Dựng hình từ kết quả thật của nghiên cứu lưới, không tính lại dữ liệu."""
    _style()
    _comparison(directory, comparison, problem)
    _convergence(directory, convergence)
    _convergence(directory, smooth, smooth=True)
    _diagnostics(directory, comparison, problem)


def check_animation_support(output_format="mp4"):
    """Kiểm tra định dạng và FFmpeg trước khi chạy lại mô phỏng."""
    if output_format not in ("mp4", "gif"):
        raise ValueError("Định dạng hoạt hình phải là 'mp4' hoặc 'gif'.")
    if output_format == "mp4" and not FFMpegWriter.isAvailable():
        raise RuntimeError("Không tìm thấy FFmpeg để xuất video MP4. Hãy cài FFmpeg và chạy lại.")


def animate_run(directory, config, problem, final_time, output_format="mp4"):
    """Chạy lại với 61 mốc thời gian và xuất MP4 cùng dữ liệu của lần chạy.

    Lịch xuất dày hơn làm thay đổi các bước rút ngắn tại mốc xuất, nên đây là
    một lần chạy riêng. Không dùng các khung hình thay số liệu của profile.pdf.
    MP4/H.264, 1920×1080, 10 khung/giây là mặc định; GIF cần chọn rõ ràng.
    """
    check_animation_support(output_format)
    _style()
    final_time = float(final_time)
    if not np.isfinite(final_time) or final_time < 0:
        raise ValueError("Thời gian hoạt hình phải hữu hạn và không âm.")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    times = [0.0] if final_time == 0 else np.linspace(0, final_time, 61)
    result, summary = stoker_run(problem, config, times)
    summary["animation_note"] = ("Lần chạy riêng với 61 mốc xuất nếu T>0. "
        "Bước thời gian được rút ngắn tại các mốc này; nghiệm có thể khác nhẹ "
        "lần chạy dùng để dựng profile.pdf.")
    save_run(directory / "animation-data", result, problem, summary)

    fig, axes = plt.subplots(2, 1, figsize=(12.8, 7.2), sharex=True, layout="constrained")
    lines, references = [], []
    reference_states = [problem.cell_averages(result.edges, float(time)) for time in result.times]
    for index, ax in enumerate(axes):
        component = 0 if index == 0 else 2
        reference_line, = ax.plot([], [], color=BLACK, linewidth=1.4,
                                  drawstyle="steps-mid", label="Stoker: trung bình ô")
        numerical_line, = ax.plot([], [], color=COLORS[config.flux], linewidth=1.7,
                                  linestyle="--" if config.flux == "rusanov" else "-",
                                  label=f"{NAMES[config.flux]}: nghiệm số")
        references.append(reference_line)
        lines.append(numerical_line)
        ax.set_ylabel(LABELS[component])
        ax.grid(True, alpha=.8)
        ax.set_xlim(*_window(problem, result))
        all_values = np.concatenate([_variables(state)[component] for state in result.states]
                                    + [_variables(state)[component] for state in reference_states])
        lower, upper = float(np.min(all_values)), float(np.max(all_values))
        padding = max(.07 * (upper - lower), .05 * abs(upper), 1e-6)
        ax.set_ylim(lower - padding, upper + padding)
    axes[0].legend(loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Tọa độ x (m)")
    video_title = f"Mô phỏng vỡ đập một chiều — {NAMES[config.flux]}, N = {config.cells}"
    title = fig.suptitle("", fontsize=13, fontweight="bold", color=RED)
    apply_figure_fonts(fig)

    def update(frame):
        numerical = _variables(result.states[frame])
        reference = _variables(reference_states[frame])
        for i, component in enumerate((0, 2)):
            lines[i].set_data(result.centers, numerical[component])
            references[i].set_data(result.centers, reference[component])
        time_text = f"{result.times[frame]:.2f}".replace(".", ",")
        title.set_text(f"{video_title}\nt = {time_text} s")
        return [*lines, *references, title]

    animation = FuncAnimation(fig, update, frames=len(result.times), interval=100, blit=False)
    if output_format == "mp4":
        writer = FFMpegWriter(fps=10, codec="libx264", bitrate=-1,
            extra_args=["-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an"],
            metadata={"title": video_title})
        dpi = 150  # 12.8×7.2 inch → 1920×1080 pixel, đúng tỉ lệ 16:9.
    else:
        writer = PillowWriter(fps=10)
        dpi = 105
    try:
        animation.save(directory / f"evolution.{output_format}", writer=writer, dpi=dpi)
    except (CalledProcessError, OSError) as error:
        raise RuntimeError(f"Không thể xuất {output_format.upper()}: {error}") from error
    finally:
        plt.close(fig)
    return summary
