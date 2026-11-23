# Hình minh họa đồ án SR5

| Hình | SVG | PNG | Mã Python |
|---|---|---|---|
| 2.1 — Kích thước và góc khớp của SR5 | [SVG](hinh_2_1.svg) |  | [hinh_2_1.py](hinh_2_1.py) |
| 2.2.1 — Sơ đồ khâu–khớp với hệ trục URDF | [SVG](hinh_2_2_1.svg) |  | [hinh_2_2_1.py](hinh_2_2_1.py) |
| 2.2.1 — Sơ đồ trục D–H của đồ án | [SVG](hinh_2_2_1_dh.svg) | [PNG](hinh_2_2_1_dh.png) |  |
| Chiều quay dương của khớp 1 | [SVG](sr5_q1_rotation.svg) | [PNG](sr5_q1_rotation.png) |  |

Trong Hình 2.1, các đường kích thước, chú thích và bảng góc khớp là đối tượng vector. Phần thân robot là ảnh raster nhúng trong SVG. Hình 2.2.1 dùng hình học vector. Chữ trong hai hình dùng phông DejaVu Sans.

Sơ đồ D–H của đồ án minh họa các chiều trục dùng trong [mã động học thuận ký hiệu](../forward_kinematics/fk_symbolic.py). Khi lập hệ trục, đặt các gốc theo hình học D–H và chọn hệ cuối tại mặt bích. Hình chiều quay khớp 1 dùng mốc x₀ trùng x₁ khi q₁ = 0.

## Vẽ lại bằng Python

Sau khi cài các phụ thuộc ROS và Python theo README của repo, chạy các lệnh sau từ thư mục gốc.

```bash
python3 fetch_sources.py
source /opt/ros/jazzy/setup.bash
/usr/bin/python3 trajectory/v_path/prepare_robot.py
/usr/bin/python3 image/hinh_2_1.py
/usr/bin/python3 image/hinh_2_2_1.py
```

Các script xuất PNG và SVG vào `image/results/`.
