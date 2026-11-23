# Động học và mô phỏng robot ROKAE SR5-5/0.9C

Mã nguồn đồ án cho tay máy nối tiếp sáu khớp quay. Hai ví dụ hàn nhận dãy vị trí và hướng mỏ theo thời gian, dùng MoveIt 2/KDL để tìm sáu góc khớp rồi kiểm tra lại bằng động học thuận. MuJoCo dùng quỹ đạo chữ V để mô phỏng chuyển động dưới tác dụng của trọng lực. Ví dụ camera lập các tư thế quan sát phôi bằng bộ giải IK riêng trong Python.

## Cấu trúc

| Thư mục | Các tệp chính và chức năng |
|---|---|
| `forward_kinematics/` | `fk.py` tính tư thế mặt bích, TCP và Jacobian hình học từ URDF. `fk_symbolic.py` tính ma trận D–H ở dạng ký hiệu. `draw_standard_dh.py` tính theo DH tiêu chuẩn và so sánh với URDF. |
| [`image/`](image/README.md) | Hình cấu tạo, sơ đồ trục D–H và minh họa chiều quay khớp 1. |
| `ik ( demo)/` | `tee.cpp` và `v_path.cpp` gọi MoveIt 2/KDL, kiểm tra giới hạn khớp và va chạm, xuất nghiệm IK theo thời gian. CMake của từng ví dụ biên dịch tệp tương ứng. |
| `trajectory/tee/` | Tạo phôi chữ T và quỹ đạo hàn thẳng. |
| `trajectory/v_path/` | Tạo phôi tấm và quỹ đạo hai đoạn chữ V. Trong mỗi ví dụ, `prepare_robot.py` tạo mô hình có mỏ, `prepare_workpiece.py` tạo phôi và các tư thế đích, `analyze.py` kiểm tra kết quả. |
| `simulation/mujoco/` | Dựng mô hình vật lý, điều khiển sáu khớp theo quỹ đạo chữ V, đo sai số bám và hiển thị robot. |
| `simulation/camera/` | Dựng cảnh camera gắn trên mặt bích, giải IK cho tư thế quan sát, mô phỏng dừng chụp và xuất các điểm chụp. |

## Chuẩn bị

Chạy các lệnh dưới đây từ thư mục chứa README. Môi trường tham chiếu là Ubuntu 24.04, ROS 2 Jazzy và Python 3.12. Sau khi cài ROS 2 Jazzy và cấu hình kho gói ROS, cài các phụ thuộc:

```bash
sudo apt install git build-essential cmake python3-venv python3-numpy python3-scipy python3-matplotlib python3-yaml ros-jazzy-moveit ros-jazzy-xacro ros-jazzy-rviz2 ros-jazzy-robot-state-publisher
python3 fetch_sources.py

/usr/bin/python3 -m venv .venv-cad
.venv-cad/bin/python -m pip install 'cadquery==2.8.0' scipy

/usr/bin/python3 -m venv .venv-mujoco
.venv-mujoco/bin/python -m pip install 'mujoco==3.12.0' numpy matplotlib
export MUJOCO_PYTHON="$PWD/.venv-mujoco/bin/python"
```

`fetch_sources.py` tải mô hình và mesh từ [ROKAE](https://github.com/RokaeRobot/rokae_ros2), cùng bộ sinh phôi T từ [Weldment Dataset](https://github.com/RoboticsNetworkTrieste/weldment-dataset). Hai mã commit được cố định trong `sources.json`. Các script ROS dùng Python hệ thống. CadQuery và MuJoCo dùng môi trường riêng.

## 1. Tạo quỹ đạo và giải động học nghịch

```bash
bash trajectory/tee/run.sh
bash trajectory/v_path/run.sh
```

Mỗi lệnh dựng robot và phôi, sinh tư thế đích, biên dịch bộ giải, chạy IK rồi đối chiếu FK và Jacobian. Dùng `--regenerate` sau khi đổi hình học phôi hoặc quy luật thời gian trong `prepare_workpiece.py`.

| Tệp trong từng ví dụ | Nội dung |
|---|---|
| `generated/targets.csv` | Thời gian (s), vị trí TCP trong hệ world (m), hướng mỏ dạng quaternion theo thứ tự qx, qy, qz, qw. |
| `results/trajectory.csv` | Thời gian (s) và sáu góc khớp (rad). |
| `results/fk.csv`, `results/jacobian.csv` | Tư thế tính lại và Jacobian tại các mẫu. |
| `results/summary.json` | Số mẫu, thời lượng, sai số, số lần vi phạm giới hạn hoặc va chạm. |

Mỏ hàn minh họa có TCP cách mặt bích 0,18 m theo trục z cục bộ. Đường chữ V dùng luật bậc năm trên từng nhánh, tốc độ đỉnh 5 mm/s và dừng 0,6 s tại đỉnh V.

## 2. Tính động học thuận riêng

Sau khi chạy ví dụ chữ V, truyền sáu góc theo độ:

```bash
/usr/bin/python3 forward_kinematics/fk.py 0 30 -60 0 -90 0
```

Với cấu hình này, vị trí mặt bích trong hệ đế xấp xỉ (643,301 mm, 136,000 mm, 595,910 mm). Chương trình in riêng ma trận tư thế mặt bích, tư thế TCP và Jacobian tại TCP. Sau khi chạy ví dụ phôi T, dùng `/usr/bin/python3 forward_kinematics/draw_standard_dh.py` để kiểm chứng bảng DH và xuất hình các hệ trục.

Để in ma trận thuần nhất từ hệ đế D–H đến hệ mặt bích ở dạng ký hiệu, cài gói `python3-sympy` nếu cần và chạy:

```bash
/usr/bin/python3 forward_kinematics/fk_symbolic.py
```

`fk_symbolic.py` dùng cách gán trục trong [sơ đồ D–H của đồ án](image/hinh_2_2_1_dh.svg). Các góc khớp theo chiều dương của URDF nên hàng 2 dùng `beta - q_2`. Khi thay số, dùng radian cho góc, giữ dấu các tham số D–H và dùng cùng một đơn vị chiều dài.

## 3. Mô phỏng và hiển thị

```bash
# Bám toàn bộ quỹ đạo chữ V bằng MuJoCo
bash simulation/mujoco/run.sh

# Mở giao diện xem chuyển động nếu có màn hình
bash simulation/mujoco/view.sh --rate 4

# Xem quỹ đạo IK trong RViz
bash trajectory/v_path/view.sh
```

Kết quả bám quỹ đạo được ghi vào `simulation/mujoco/results/tracking_summary.json`. Sau khi chạy cả hai ví dụ hàn và dựng cảnh MuJoCo, chạy mô phỏng camera:

```bash
bash simulation/camera/run.sh --part all
```

Mỗi phôi có một thư mục kết quả riêng trong `simulation/camera/results/`. Tệp `camera_waypoints.csv` chứa tư thế camera tại thời điểm chụp, với quaternion theo thứ tự qw, qx, qy, qz. Camera và gá lắp là mô hình giả định của đồ án.

## Đưa lên GitHub

Dùng thư mục chứa README này làm thư mục gốc của repository. Tệp `.gitignore` loại môi trường Python, nguồn tải ngoài, thư mục build, tệp sinh tự động và kết quả chạy. Sau khi clone sang máy khác, thực hiện lại bước chuẩn bị để các đường dẫn mesh được tạo đúng theo vị trí mới.
