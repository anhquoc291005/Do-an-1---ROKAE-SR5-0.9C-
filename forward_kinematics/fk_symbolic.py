#!/usr/bin/env python3
"""Động học thuận ký hiệu của SR5 theo bảng D–H của hình đồ án.

Các q_i theo chiều dương của URDF. Góc tính bằng radian. Dùng cùng
đơn vị cho mọi chiều dài và giữ dấu của a_3, d_5 khi thay số.
"""

import sympy as sp


q_1, q_2, q_3, q_4, q_5, q_6 = sp.symbols("q_1:7", real=True)
beta, delta_1, delta_6 = sp.symbols("beta delta_1 delta_6", real=True)
d_1, d_4, d_5, d_6, a_2, a_3 = sp.symbols("d_1 d_4 d_5 d_6 a_2 a_3", real=True)


def dh_transform(theta, d, a, alpha):
    """Lập ma trận D–H tiêu chuẩn từ hệ i đến hệ i−1."""
    cos_theta, sin_theta = sp.cos(theta), sp.sin(theta)
    cos_alpha, sin_alpha = sp.cos(alpha), sp.sin(alpha)
    return sp.Matrix([
        [cos_theta, -sin_theta * cos_alpha, sin_theta * sin_alpha, a * cos_theta],
        [sin_theta, cos_theta * cos_alpha, -cos_theta * sin_alpha, a * sin_theta],
        [0, sin_alpha, cos_alpha, d],
        [0, 0, 0, 1],
    ])


def symbolic_fk():
    """Trả về ma trận 4×4 từ hệ mặt bích {6} sang hệ đế D–H {0}."""
    half_pi = sp.pi / 2
    dh_rows = (
        (q_1 + delta_1, d_1, 0, half_pi),
        (beta - q_2, 0, a_2, 0),
        (q_3 - beta, 0, a_3, -half_pi),
        (q_4, d_4, 0, half_pi),
        (q_5 + sp.pi, d_5, 0, half_pi),
        (q_6 + delta_6, d_6, 0, 0),
    )
    A1, A2, A3, A4, A5, A6 = (dh_transform(*row) for row in dh_rows)

    # α₂ = 0 nên có thể gộp góc θ₂ + θ₃ = q₃ − q₂.
    A23 = (A2 * A3).applyfunc(lambda entry: sp.trigsimp(entry, method="fu"))
    return A1 * (A23 * (A4 * (A5 * A6)))


if __name__ == "__main__":
    T_06 = symbolic_fk()
    print("T_06 = Matrix([")
    for row in T_06.tolist():
        print("    [")
        for entry in row:
            print(f"        {entry},")
        print("    ],")
    print("])")
