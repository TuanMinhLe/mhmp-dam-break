# Mô phỏng vỡ đập một chiều

Notebook của Nhóm 7 cho học phần Mô hình mô phỏng: giải hệ nước nông bằng phương pháp thể tích hữu hạn bậc một với thông lượng **Rusanov** và **HLL**, đối chiếu với nghiệm **Stoker**.

Kho mã chứa một notebook, mã Python, dữ liệu CSV dùng để kiểm tra kết quả và các kiểm thử tính toán. Thư mục **[outputs/](outputs/)** kèm sẵn kết quả của lần chạy với cấu hình ban đầu, để mọi người có thể xem trước mà không cần cài Python.

## Xem kết quả đã chạy

Mở các hình PNG, bảng CSV và tệp JSON ngay trên GitHub. Với **[hoạt ảnh tương tác](outputs/mo-phong-vo-dap.html)**, tải tệp HTML về máy rồi mở bằng trình duyệt và nhấn **▶ Phát**. 

Bộ kết quả gồm 10 tệp:

| Tệp | Nội dung |
|---|---|
| [mo-phong-vo-dap.html](outputs/mo-phong-vo-dap.html) | Hoạt ảnh 0–6 s, có phát, tạm dừng, thanh thời gian và lựa chọn HLL/Rusanov. |
| [nghiem-so-va-stoker.png](outputs/nghiem-so-va-stoker.png) | So sánh nghiệm số với nghiệm Stoker. |
| [can-bang.png](outputs/can-bang.png) | Kiểm tra cân bằng có thông lượng biên. |
| [anh-huong-luoi.png](outputs/anh-huong-luoi.png) | Ảnh hưởng của số ô lưới. |
| [anh-huong-cfl.png](outputs/anh-huong-cfl.png) | Ảnh hưởng của CFL. |
| [anh-huong-ti-so-do-sau.png](outputs/anh-huong-ti-so-do-sau.png) | Ảnh hưởng của tỉ số độ sâu. |
| [thang-do-sau.png](outputs/thang-do-sau.png) | So sánh khi thay đổi thang độ sâu tuyệt đối. |
| [bao-cao-72-to-hop.png](outputs/bao-cao-72-to-hop.png) | Bảng màu sai số của 72 tổ hợp trong dữ liệu đã lưu. |
| [khao-sat-trong-notebook.csv](outputs/khao-sat-trong-notebook.csv) | Kết quả 24 lượt tính mới theo lịch lưu thưa. |
| [tom-tat-phien-chay.json](outputs/tom-tat-phien-chay.json) | Tham số, số lượt tính và các chỉ số kiểm tra của phiên chạy. |

Muốn đổi tham số hoặc tính lại kết quả, làm theo hướng dẫn cài đặt bên dưới. Chạy lại notebook sẽ cập nhật các tệp cùng tên trong `outputs/`.

## Cài đặt và mở notebook

Dùng Python **3.11 trở lên**. Từ terminal:

```bash
git clone https://github.com/TuanMinhLe/mhmp-dam-break.git
cd mhmp-dam-break
python -m venv .venv
```

Kích hoạt môi trường:

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Sau đó cài thư viện và mở JupyterLab:

```bash
python -m pip install -r requirements.txt
python -m jupyterlab
```

Mở **[bao-cao-nhom-7.ipynb](bao-cao-nhom-7.ipynb)**, chọn kernel Python của môi trường vừa cài và chọn **Restart Kernel and Run All Cells**. Đợi phần tính toán chạy xong, rồi nhấn **▶ Phát** một lần trong đồ thị tương tác. Không cần chạy lại ô cho từng bước thời gian.

## Cấu hình ban đầu

| Đại lượng | Giá trị |
|---|---:|
| Miền tính | $[-5,5]$ m |
| Vị trí đập | $x_d=0$ |
| Gia tốc trọng trường | $g=9{,}81$ m/s² |
| Độ sâu thượng lưu / hạ lưu | 5 mm / 1 mm |
| Số ô | 400 |
| CFL | 0,45 |
| Thời gian vật lý | 0–6 s |
| Số trạng thái lưu cho hoạt ảnh | 121 |
| Thời gian phát trên màn hình | khoảng 10 s |

Sửa ô tham số để thử số ô, độ sâu, CFL hoặc thời gian khác, rồi chạy lại toàn bộ. Cần giữ $h_L>h_R>0$, $0<\mathrm{CFL}\leq0{,}5$ và thời điểm đối chiếu trước khi sóng chạm biên; notebook kiểm tra các điều kiện này.

Các nút **Tạm dừng**, **Về đầu**, thanh thời gian và lựa chọn HLL/Rusanov dùng cùng các trạng thái đã tính. Thời gian phát không làm thay đổi bước thời gian của bộ giải. Sau khi chạy, có thể mở `outputs/mo-phong-vo-dap.html` bằng trình duyệt.

## Nội dung tính toán

- Mô hình nước nông, nghiệm Stoker và công thức thông lượng Rusanov/HLL.
- Hoạt ảnh của hai nghiệm số cùng trung bình ô Stoker.
- Sai số $L^1$ chuẩn hóa, độ sâu nhỏ nhất và cân bằng có thông lượng biên.
- Khảo sát riêng số ô, CFL, tỉ số độ sâu và thang độ sâu tuyệt đối.
- Đọc bộ dữ liệu 84 lượt đã lưu; tùy chọn chạy lại 42 lượt cơ sở và 84 lượt bổ sung.

Theo mặc định, notebook tính mới hai lượt cho hoạt ảnh và 24 lượt theo lịch lưu thưa. Tùy chọn `RUN_FULL_REPORT_STUDY` được tắt. Nếu bật, dữ liệu đầy đủ được tạo trong `outputs/full-report-rerun/`.

Lịch lưu của hoạt ảnh gồm 121 mốc, trong khi các phép đo theo cấu hình báo cáo dùng các mốc `[0,2,4,6]` s. Bước cuối mỗi khoảng bị rút ngắn để đến đúng mốc lưu, nên hai lịch có thể cho sai số hơi khác nhau. Không dùng số liệu trên hoạt ảnh thay cho bảng sai số của khảo sát.

## Cấu trúc

```text
bao-cao-nhom-7.ipynb     Notebook duy nhất cần mở
parameter_study.py       Khảo sát kết hợp 84 lượt
dambreak/                Bộ giải, nghiệm tham chiếu và hàm vẽ
checks/source-integrity.json
                         SHA-256 của mã và các tệp dữ liệu kèm theo
data/report-baseline/    Sai số theo lưới của bộ cơ sở
data/report-parameters/  Dữ liệu khảo sát kết hợp và các giá trị tổng hợp
outputs/                HTML tương tác, 7 hình PNG, CSV và JSON đã chạy sẵn
tests/                   Kiểm thử nghiệm tham chiếu, thông lượng và bộ giải
tools/clear_notebook_outputs.py
                         Xóa đầu ra notebook trước khi commit
```

Bốn tệp dữ liệu được giữ vì notebook đọc trực tiếp:

| Tệp | Mục đích |
|---|---|
| `data/report-baseline/convergence.csv` | So sai số của hai lượt chuẩn vừa chạy với dữ liệu đã lưu. |
| `data/report-parameters/factorial.csv` | Vẽ bảng màu của 72 tổ hợp lưới, CFL, tỉ số độ sâu và thông lượng. |
| `data/report-parameters/all-runs.csv` | Kiểm tra số lượt trong bộ 84 lượt. |
| `data/report-parameters/summary.json` | Đọc danh sách tham số, số cấu hình và các cực trị để hiển thị. |

Notebook chỉ đọc các tệp trong `data/`; kết quả tính mới nằm trong `outputs/`. Dấu SHA-256 phát hiện thay đổi ở mã hoặc dữ liệu gốc. Nếu chủ động sửa thuật toán, cần kiểm thử lại rồi cập nhật danh sách dấu kiểm tương ứng; sửa ô tham số của notebook không làm thay đổi danh sách này.

## Kiểm tra mã

```bash
python -m unittest discover -s tests -v
```

Bộ kiểm thử gồm 32 phép thử đã có của dự án: điều kiện Rankine–Hugoniot, tích phân tham chiếu, tính nhất quán thông lượng, cân bằng, độ sâu dương, dữ liệu đầu vào và hội tụ trên nghiệm trơn. Không cần trình duyệt để chạy bộ này.

Trước khi commit notebook đã chạy:

```bash
python tools/clear_notebook_outputs.py
```

Lệnh này chỉ xóa đầu ra trong các ô notebook, không xóa tệp ở `outputs/`. `.gitignore` cho phép lưu đúng 10 tệp kết quả nêu trên; các đầu ra khác, thư mục `outputs/full-report-rerun/`, môi trường Python và bộ nhớ đệm được loại khỏi Git. Sau khi thử tham số khác, kiểm tra các tệp kết quả thay đổi trước khi commit để phần mô tả và cấu hình được lưu cùng nhau.

Môi trường đã dùng để kiểm tra: Python 3.14.4, NumPy 2.5.4, Matplotlib 3.11.2 và Plotly 7.1.0. `requirements.txt` cho phép các phiên bản tương thích; khi đổi môi trường, chạy lại bộ kiểm thử trước khi đối chiếu số liệu.

## Phạm vi

Bộ giải xét dòng chảy một chiều trên đáy phẳng, không ma sát và hạ lưu có nước. Kết quả là kiểm chứng số cho mô hình, chưa phải đối chiếu với thí nghiệm vỡ đập thực tế. Chưa xử lý nền khô hoặc địa hình biến đổi.

## Tài liệu tham khảo

1. David I. Ketcheson, Randall J. LeVeque, and Mauricio J. del Razo (2020). *Riemann Problems and Jupyter Solutions: Theory and Approximate Solvers for Hyperbolic PDEs*. SIAM. [Shallow water chapter](https://www.clawpack.org/riemann_book/html/Shallow_water.html).
2. Randall J. LeVeque (2002). *Finite Volume Methods for Hyperbolic Problems*. Cambridge University Press. [DOI](https://doi.org/10.1017/CBO9780511791253).
3. Amiram Harten, Peter D. Lax, and Bram van Leer (1983). *On Upstream Differencing and Godunov-Type Schemes for Hyperbolic Conservation Laws*. SIAM Review, 25(1), 35–61. [DOI](https://doi.org/10.1137/1025002).
