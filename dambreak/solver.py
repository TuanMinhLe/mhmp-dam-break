"""Bộ giải PDE độc lập với nghiệm Stoker: Rusanov/HLL, Euler bậc một.

U[:, 0] là trung bình ô của h [m]; U[:, 1] là trung bình ô của q [m²/s].
Chỉ hỗ trợ h>0, đáy phẳng, không ma sát; không cắt/sửa độ sâu âm.
"""
from dataclasses import dataclass
import math
import numpy as np


def _positive_number(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} phải hữu hạn và dương.")


def _state(values):
    U = np.asarray(values, dtype=float)
    if U.ndim < 1 or U.shape[-1] != 2 or not np.all(np.isfinite(U)):
        raise ValueError("Trạng thái phải có chiều cuối bằng 2 và các giá trị hữu hạn.")
    if np.any(U[..., 0] <= 0):
        raise ValueError("Bộ giải nền ướt yêu cầu h > 0; không dùng ngưỡng cắt độ sâu.")
    return U


def physical_flux(U, gravity=9.81):
    """F(U)=(q, q²/h + gh²/2), cùng hình dạng với U."""
    _positive_number(gravity, "g")
    U = _state(U)
    h, q = U[..., 0], U[..., 1]
    return np.stack((q, q * (q / h) + 0.5 * gravity * h**2), axis=-1)


def numerical_flux(left, right, gravity=9.81, method="hll"):
    """Thông lượng mặt ô và tốc độ lớn nhất dùng để chọn bước thời gian.

    HLL dùng cực trị u±sqrt(gh) tại HAI trạng thái, kèm 0.
    Đây là ước lượng sóng của bộ giải xấp xỉ, không phải tốc độ Stoker.
    """
    _positive_number(gravity, "g")
    left, right = np.broadcast_arrays(_state(left), _state(right))
    fl, fr = physical_flux(left, gravity), physical_flux(right, gravity)
    ul, ur = left[..., 1] / left[..., 0], right[..., 1] / right[..., 0]
    cl, cr = np.sqrt(gravity * left[..., 0]), np.sqrt(gravity * right[..., 0])
    if method == "rusanov":
        speed = np.maximum(np.abs(ul) + cl, np.abs(ur) + cr)
        flux = 0.5 * (fl + fr) - 0.5 * speed[..., None] * (right - left)
    elif method == "hll":
        sl = np.minimum(0.0, np.minimum(ul - cl, ur - cr))
        sr = np.maximum(0.0, np.maximum(ul + cl, ur + cr))
        # Vì h>0, luôn có sr-sl>0. Công thức cũng đúng nếu sl=0 hoặc sr=0.
        flux = (sr[..., None] * fl - sl[..., None] * fr
                + (sl * sr)[..., None] * (right - left)) / (sr - sl)[..., None]
        speed = np.maximum(-sl, sr)
    else:
        raise ValueError("Thông lượng phải là 'rusanov' hoặc 'hll'.")
    if not np.all(np.isfinite(flux)) or not np.all(np.isfinite(speed)):
        raise FloatingPointError("Thông lượng bị tràn số; kiểm tra thang tham số.")
    return flux, speed


@dataclass(frozen=True)
class FVConfig:
    x_min: float = -5.0
    x_max: float = 5.0
    cells: int = 400
    gravity: float = 9.81
    cfl: float = 0.45
    flux: str = "hll"
    boundary: str = "outflow"
    max_steps: int = 1_000_000

    def __post_init__(self):
        if not (np.isfinite(self.x_min) and np.isfinite(self.x_max)
                and self.x_max > self.x_min):
            raise ValueError("Miền tính phải hữu hạn và x_max > x_min.")
        if isinstance(self.cells, bool) or not isinstance(self.cells, (int, np.integer)) or self.cells < 2:
            raise ValueError("Số ô phải là số nguyên >= 2.")
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, (int, np.integer)) or self.max_steps < 1:
            raise ValueError("max_steps phải là số nguyên dương.")
        _positive_number(self.gravity, "g")
        if not np.isfinite(self.cfl) or not 0 < self.cfl <= 0.5:
            raise ValueError("Phiên bản bậc một này yêu cầu 0 < CFL <= 0.5.")
        if self.flux not in ("rusanov", "hll"):
            raise ValueError("Thông lượng không được hỗ trợ.")
        if self.boundary not in ("outflow", "periodic"):
            raise ValueError("Biên phải là 'outflow' hoặc 'periodic'.")

    @property
    def edges(self):
        return np.linspace(self.x_min, self.x_max, self.cells + 1)

    @property
    def dx(self):
        return (self.x_max - self.x_min) / self.cells


@dataclass
class SimulationResult:
    edges: np.ndarray
    times: np.ndarray
    states: np.ndarray
    history: list
    config: FVConfig

    @property
    def centers(self):
        return 0.5 * (self.edges[:-1] + self.edges[1:])


def solve(initial, config, output_times):
    """Tiến PDE từ t=0 và lưu trạng thái đúng các thời điểm yêu cầu.

    Một ô ảo ở mỗi đầu: sao chép ô trong (outflow) hoặc nối vòng (periodic).
    Nhật ký dùng CHÍNH thông lượng biên đã cập nhật, kể cả bước bị rút ngắn.
    """
    U = _state(initial).copy()
    if U.shape != (config.cells, 2):
        raise ValueError(f"Dữ liệu đầu phải có dạng ({config.cells}, 2).")
    times = np.asarray(output_times, dtype=float)
    if (times.ndim != 1 or times.size == 0 or not np.all(np.isfinite(times))
            or np.any(times < 0) or np.any(np.diff(times) <= 0)):
        raise ValueError("Thời điểm xuất phải hữu hạn, không âm, tăng nghiêm ngặt.")
    dx, g = config.dx, config.gravity
    initial_total = dx * np.sum(U, axis=0)
    cumulative = np.zeros(2)
    compensation = np.zeros(2)  # Cộng có bù để giảm sai số nhật ký tích lũy.
    t, step = 0.0, 0
    history, snapshots = [], []

    def record(dt, speed, left_flux, right_flux):
        total = dx * np.sum(U, axis=0)
        residual = total - initial_total - cumulative
        energy = dx * np.sum(0.5 * U[:, 1]**2 / U[:, 0] + 0.5 * g * U[:, 0]**2)
        history.append(dict(step=step, time=t, dt=dt, max_speed=speed,
            cfl=dt * speed / dx, min_depth=float(np.min(U[:, 0])),
            mass=float(total[0]), momentum=float(total[1]), energy=float(energy),
            mass_residual=float(residual[0]), momentum_residual=float(residual[1]),
            cumulative_mass_inflow=float(cumulative[0]),
            cumulative_momentum_inflow=float(cumulative[1]),
            flux_left_h=float(left_flux[0]), flux_left_q=float(left_flux[1]),
            flux_right_h=float(right_flux[0]), flux_right_q=float(right_flux[1])))

    record(0.0, 0.0, np.zeros(2), np.zeros(2))
    for target in times:
        while t < target:
            if step >= config.max_steps:
                raise RuntimeError("Vượt max_steps; kiểm tra thời gian, lưới và tham số.")
            if config.boundary == "periodic":
                padded = np.concatenate((U[-1:], U, U[:1]))
            else:
                padded = np.concatenate((U[:1], U, U[-1:]))
            faces, speeds = numerical_flux(padded[:-1], padded[1:], g, config.flux)
            speed = float(np.max(speeds))
            remaining = float(target - t)
            dt = min(config.cfl * dx / speed, remaining)
            if not math.isfinite(dt) or dt <= 0 or t + dt <= t:
                raise FloatingPointError("Bước thời gian quá nhỏ hoặc không hữu hạn.")
            next_U = U - (dt / dx) * (faces[1:] - faces[:-1])
            if not np.all(np.isfinite(next_U)) or np.any(next_U[:, 0] <= 0):
                raise FloatingPointError("Bước cập nhật không còn trạng thái ướt hữu hạn; không sửa/cắt nghiệm.")
            increment = dt * (faces[0] - faces[-1])
            corrected = increment - compensation
            updated = cumulative + corrected
            compensation = (updated - cumulative) - corrected
            cumulative = updated
            U = next_U
            t = float(target) if dt == remaining else t + dt
            step += 1
            record(dt, speed, faces[0], faces[-1])
        snapshots.append(U.copy())
    return SimulationResult(config.edges, times.copy(), np.stack(snapshots), history, config)
