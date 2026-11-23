# Hình minh họa đồ án SR5

| Hình | Tệp mở trong Adobe Illustrator | Mã Python |
|---|---|---|
| 2.1 — Kích thước và góc khớp của SR5 | [hinh_2_1.svg](hinh_2_1.svg) | [hinh_2_1.py](hinh_2_1.py) |
| 2.2.1 — Sơ đồ khâu–khớp và các trục x, z | [hinh_2_2_1.svg](hinh_2_2_1.svg) | [hinh_2_2_1.py](hinh_2_2_1.py) |

Trong Hình 2.1, các đường kích thước, chú thích và bảng góc khớp là đối tượng vector. Phần thân robot là ảnh raster nhúng trong SVG. Hình 2.2.1 dùng hình học vector. Chữ trong hai hình dùng phông DejaVu Sans.

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
