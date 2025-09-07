# 问题一

import numpy as np
import math

# 初始参数设置
M1_0_position = np.array([20000.0, 0.0, 2000.0])
FY1_0_position = np.array([17800.0, 0.0, 1800.0])
v_fy1 = 120  # m/s
v_missile = 300  # m/s
fake_target = np.array([0.0, 0.0, 0.0])  # 原点(假目标)
true_target = np.array([0.0, 200.0, 0.0])  # 真目标
delayed_release = 1.5  # s
delayed_detonation = 3.6  # s
effective_radius = 10  # 烟雾有效半径 m
effective_time = 20  # s
sinking_speed = 3  # m/s
g = 9.8  # 重力加速度m/s²

# 圆柱面参数
y0 = 200
r0 = 7
h0 = 10

dt = 0.1

# 1.5s时无人机投下烟雾弹时的位置
FY1_1_5_position = np.array([
    FY1_0_position[0] - v_fy1 * 1.5,
    FY1_0_position[1],
    FY1_0_position[2]
])

# 5.1s时烟雾弹爆炸前的位置
FY11_5_1_position = np.array([
    FY1_1_5_position[0] - v_fy1 * 3.6,
    FY1_1_5_position[1],
    FY1_1_5_position[2] - g * 3.6 ** 2 / 2
])

M1_to_fake_vec = fake_target - M1_0_position
M1_sum = np.linalg.norm(M1_to_fake_vec)
M1_dir_unit = M1_to_fake_vec / M1_sum

# 定义所有测试点
angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
heights = [0, h0 / 2, h0]


def get_valid_point(angle, z):
    x1 = r0 * np.cos(angle)
    y1 = y0 + r0 * np.sin(angle)
    z1 = z
    return [x1, y1, z1]


def get_all():
    """获取所有测试点"""
    test_points = []
    for angle in angles:
        for z in heights:
            test_points.append(get_valid_point(angle, z))
    return test_points


def data(vars, M1_pos, FY11_pos):
    x1, y1, z1 = vars
    x_M1_t, y_M1_t, z_M1_t = M1_pos
    x_FY11_t, y_FY11_t, z_FY11_t = FY11_pos

    term1 = (x_M1_t - x1) * (x1 - x_FY11_t)
    term2 = (y_M1_t - y1) * (y1 - y_FY11_t)
    term3 = (z_M1_t - z1) * (z1 - z_FY11_t)
    sum_terms = term1 + term2 + term3
    part1 = 4 * math.pow(sum_terms, 2)

    squared_sum1 = math.pow(x_M1_t - x1, 2) + math.pow(y_M1_t - y1, 2) + math.pow(z_M1_t - z1, 2)
    squared_sum2 = math.pow(x1 - x_FY11_t, 2) + math.pow(y1 - y_FY11_t, 2) + math.pow(z1 - z_FY11_t, 2) - math.pow(
        effective_radius, 2)
    part2 = 4 * squared_sum1 * squared_sum2

    delta = part1 - part2
    return delta, squared_sum1, 2 * sum_terms


def valid_for_point(delta, a, b):
    if delta < 0:
        return False
    k1 = (-b + np.sqrt(delta)) / (2 * a)
    k2 = (-b - np.sqrt(delta)) / (2 * a)
    return (0 <= k1 <= 1) or (0 <= k2 <= 1)


def check_validity_all_points(t, M1_pos, FY11_pos):
    test_points = get_all()

    for test_point in test_points:
        delta, a, b = data(test_point, M1_pos, FY11_pos)
        if not (delta >= 0 and valid_for_point(delta, a, b)):
            return False

    return True


def check(t):
    if t < 5.1:
        return False

    M1_pos = M1_0_position + v_missile * t * M1_dir_unit
    FY11_pos = np.array([
        FY11_5_1_position[0],
        FY11_5_1_position[1],
        FY11_5_1_position[2] - sinking_speed * (t - 5.1)
    ])

    return check_validity_all_points(t, M1_pos, FY11_pos)

record = []

for t in np.arange(6.0, 10.0, dt):
    M1_t_position = M1_0_position + v_missile * t * M1_dir_unit

    if t < 5.1:
        continue
    FY11_t_position = np.array([
        FY11_5_1_position[0],
        FY11_5_1_position[1],
        FY11_5_1_position[2] - sinking_speed * (t - 5.1)
    ])

    valid_flag = False
    for angle in angles:
        for z in heights:
            test_point = get_valid_point(angle, z)
            delta, a, b = data(test_point, M1_t_position, FY11_t_position)
            if delta >= 0 and valid_for_point(delta, a, b):
                valid_flag = True
                break
        if valid_flag:
            break

    if valid_flag:
        record.append(t)

if record:
    def binary_search_earliest(t_low, t_high):
        tol = 1e-6
        max_iter = 100

        valid_low = check(t_low)
        valid_high = check(t_high)

        if valid_low:
            return t_low

        if not valid_high:
            return None

        for iteration in range(max_iter):
            if abs(t_high - t_low) < tol:
                return t_high

            t_mid = (t_low + t_high) / 2
            valid_mid = check(t_mid)

            if valid_mid:
                t_high = t_mid
            else:
                t_low = t_mid

        return t_high


    def binary_search_latest(t_low, t_high):
        tol = 1e-6
        max_iter = 100

        valid_low = check(t_low)
        valid_high = check(t_high)

        if valid_high:
            return t_high

        if not valid_low:
            return None

        for iteration in range(max_iter):
            if abs(t_high - t_low) < tol:
                return t_low

            t_mid = (t_low + t_high) / 2
            valid_mid = check(t_mid)

            if valid_mid:
                t_low = t_mid
            else:
                t_high = t_mid

        return t_low

    if record:
        earliest_value = min(record)
        latest_values = max(record)
        search_margin = 0.5

        earliest_search_low = max(5.1, earliest_value - search_margin)
        earliest_search_high = earliest_value + 0.2
        pea = binary_search_earliest(earliest_search_low, earliest_search_high)

        latest_search_low = latest_values - 0.2
        latest_search_high = latest_values + search_margin
        pla = binary_search_latest(latest_search_low, latest_search_high)

    test_points = get_all()

    individual_windows = []

    for i, test_point in enumerate(test_points):
        angle_idx = i // len(heights)
        height_idx = i % len(heights)

        point_record = []
        for t in np.arange(6.0, 10.0, dt):
            if t < 5.1:
                continue

            M1_pos = M1_0_position + v_missile * t * M1_dir_unit
            FY11_pos = np.array([
                FY11_5_1_position[0],
                FY11_5_1_position[1],
                FY11_5_1_position[2] - sinking_speed * (t - 5.1)
            ])

            delta, a, b = data(test_point, M1_pos, FY11_pos)
            if delta >= 0 and valid_for_point(delta, a, b):
                point_record.append(t)

        if point_record:
            earliest = min(point_record)
            latest = max(point_record)
            duration = latest - earliest
            individual_windows.append((earliest, latest, duration))
        else:
            individual_windows.append((None, None, 0))

    # 找到最短的时间窗口
    valid_windows = [(start, end, duration) for start, end, duration in individual_windows if start is not None]
    if valid_windows:
        shortest_window = min(valid_windows, key=lambda x: x[2])
        longest_window = max(valid_windows, key=lambda x: x[2])

        if pea is not None and pla is not None:
            all_points_window = pla - pea
            print(f"所有点同时被遮蔽的窗口长度: {all_points_window:.6f}s")