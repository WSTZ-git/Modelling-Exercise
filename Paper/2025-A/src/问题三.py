# 问题三

import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, pi
import random
import math

plt.rc("font", family='DengXian')
plt.rcParams['axes.unicode_minus'] = False

M1_0_position = np.array([20000.0, 0.0, 2000.0])
FY1_0_position = np.array([17800.0, 0.0, 1800.0])
v_missile = 300  # m/s
fake_target = np.array([0.0, 0.0, 0.0])  # 原点(假目标)
true_target = np.array([0.0, 200.0, 0.0])  # 真目标
effective_radius = 10  # m
effective_time = 20  # s
sinking_speed = 3  # m/s
g = 9.8  # 重力加速度m/s
r0 = 7
h0 = 10
y0 = 200

def obj(t, target, FY1i_t_position):
    unit_dir = -M1_0_position / np.linalg.norm(M1_0_position)
    M1_t_position = M1_0_position + v_missile * t * unit_dir
    OM1_length = np.linalg.norm(M1_t_position)

    x1, y1, z1 = target
    x_M1_t, y_M1_t, z_M1_t = M1_t_position
    x_FY11_t, y_FY11_t, z_FY11_t = FY1i_t_position

    term1 = (x_M1_t - x1) * (x1 - x_FY11_t)
    term2 = (y_M1_t - y1) * (y1 - y_FY11_t)
    term3 = (z_M1_t - z1) * (z1 - z_FY11_t)
    dot_product = term1 + term2 + term3
    part1 = 4 * (dot_product ** 2)

    a_sq = (x_M1_t - x1) ** 2 + (y_M1_t - y1) ** 2 + (z_M1_t - z1) ** 2
    b_sq = (x1 - x_FY11_t) ** 2 + (y1 - y_FY11_t) ** 2 + (z1 - z_FY11_t) ** 2
    part2 = 4 * a_sq * (b_sq - effective_radius ** 2)

    delta = part1 - part2
    if delta < 0:
        return 0, False

    b = 2 * dot_product
    k1 = (-b + np.sqrt(delta)) / (2 * a_sq)
    k2 = (-b - np.sqrt(delta)) / (2 * a_sq)
    d1 = np.sqrt(pow(k1 * (x_M1_t - x1) + x1, 2) + pow(k1 * (y_M1_t - y1) + y1, 2) + pow(k1 * (z_M1_t - z1) + z1, 2))
    d2 = np.sqrt(pow(k2 * (x_M1_t - x1) + x1, 2) + pow(k2 * (y_M1_t - y1) + y1, 2) + pow(k2 * (z_M1_t - z1) + z1, 2))
    check = d1 <= OM1_length or d2 <= OM1_length

    return delta, check


def calculate_bomb_position(alpha, v, ta, tb, t):
    FY1_ta_position = np.array([
        FY1_0_position[0] + v * ta * cos(alpha),
        FY1_0_position[1] + v * ta * sin(alpha),
        FY1_0_position[2]
    ])

    FY1_tb_position = np.array([
        FY1_ta_position[0] + v * (tb - ta) * cos(alpha),
        FY1_ta_position[1] + v * (tb - ta) * sin(alpha),
        FY1_ta_position[2] - g * (tb - ta) ** 2 / 2
    ])

    if t >= tb:
        FY1_t_position = np.array([
            FY1_tb_position[0],
            FY1_tb_position[1],
            FY1_tb_position[2] - sinking_speed * (t - tb)
        ])
    else:
        FY1_t_position = np.array([
            FY1_ta_position[0] + v * (t - ta) * cos(alpha),
            FY1_ta_position[1] + v * (t - ta) * sin(alpha),
            FY1_ta_position[2] - g * (t - ta) ** 2 / 2
        ])

    return FY1_t_position

def check_constraints(params):
    alpha, v, t1a, t2a, t3a, t1b, t2b, t3b = params

    if v < 70 or v > 140:
        return False

    if not (0 < t1a < t2a < t3a and t2a - t1a >= 1 and t3a - t2a >= 1):
        return False

    if not (t1b > t1a and t2b > t2a and t3b > t3a):
        return False

    if alpha < 0 or alpha > 2 * pi:
        return False

    return True

def function(params, debug=False):
    if not check_constraints(params):
        return -1

    alpha, v, t1a, t2a, t3a, t1b, t2b, t3b = params
    time_step = 0.01
    target = get_valid_initial_point()

    missile_time_to_target = np.linalg.norm(M1_0_position) / v_missile

    covered_time_points = []
    coverage_details = []

    bomb_params = [(t1a, t1b), (t2a, t2b), (t3a, t3b)]

    for t in np.arange(0, min(missile_time_to_target, 100), time_step):
        is_covered = False
        covering_bombs = []

        for bomb_idx, (ta, tb) in enumerate(bomb_params):
            if tb <= t <= tb + effective_time:
                bomb_position = calculate_bomb_position(alpha, v, ta, tb, t)
                delta, check = obj(t, target, bomb_position)

                if delta > 0 and check:
                    covering_bombs.append(bomb_idx + 1)
                    if not is_covered:
                        is_covered = True

        if is_covered:
            covered_time_points.append(t)
            if debug:
                coverage_details.append({
                    'time': t,
                    'covering_bombs': covering_bombs,
                    'count': len(covering_bombs)
                })

    total_coverage_time = total_coverage(covered_time_points, time_step)

    return total_coverage_time

def get_valid_initial_point():
    angle = np.pi / 4
    x1_init = r0 * np.cos(angle)
    y1_init = y0 + r0 * np.sin(angle)
    z1_init = h0 / 2
    return [x1_init, y1_init, z1_init]

def total_coverage(time_points, dt):
    if len(time_points) == 0:
        return 0

    if len(time_points) == 1:
        return dt

    simple_total = len(time_points) * dt

    return simple_total


def analyze_bomb_contributions(params):
    if not check_constraints(params):
        return

    alpha, v, t1a, t2a, t3a, t1b, t2b, t3b = params
    time_step = 0.1
    target = get_valid_initial_point()
    missile_time_to_target = np.linalg.norm(M1_0_position) / v_missile

    bomb_params = [(t1a, t1b), (t2a, t2b), (t3a, t3b)]

    individual_contributions = []

    for bomb_idx, (ta, tb) in enumerate(bomb_params):
        covered_times = []

        for t in np.arange(0, min(missile_time_to_target, 100), time_step):
            if tb <= t <= tb + effective_time:
                bomb_position = calculate_bomb_position(alpha, v, ta, tb, t)
                delta, check = obj(t, target, bomb_position)

                if delta > 0 and check:
                    covered_times.append(t)

        contribution = len(covered_times) * time_step
        individual_contributions.append(contribution)

    total = function(params, debug=False)

    return individual_contributions, total

def objective_function(params):
    return function(params, debug=False)


def adaptive_simulated_annealing(initial_params=None, max_iterations=10000, initial_temp=100, final_temp=0.01,
                                 restart_threshold=500):
    param_ranges = [
        (0, 2 * pi),
        (70, 140),
        (0.1, 5),
        (1.1, 6),
        (3.1, 8),
        (0.2, 10),
        (2.2, 12),
        (4.2, 14)
    ]

    # 初始参数
    if initial_params is None:
        initial_params = np.array([
            3.1356, 140, 1.05, 3.61, 4.91, 5.33, 8.9, 10.78
        ])

    current_params = initial_params.copy()
    current_score = objective_function(current_params)
    best_params = current_params.copy()
    best_score = current_score
    best_scores = [best_score]

    history_best = [(current_params.copy(), current_score)]

    temp = initial_temp
    no_improve_count = 0
    iteration = 0

    cooling_rate = (final_temp / initial_temp) ** (1 / max_iterations)

    while iteration < max_iterations and temp > final_temp:
        scale = 0.5 * (1 + temp / initial_temp)

        new_params = current_params.copy()
        param_index = random.randint(0, len(new_params) - 1)

        if param_index == 0:
            range_width = param_ranges[param_index][1] - param_ranges[param_index][0]
            new_params[param_index] += random.uniform(-0.05 * scale, 0.05 * scale) * range_width
        elif param_index == 1:
            new_params[param_index] += random.uniform(-3 * scale, 3 * scale)
        else:
            range_width = param_ranges[param_index][1] - param_ranges[param_index][0]
            new_params[param_index] += random.uniform(-0.3 * scale, 0.3 * scale) * range_width


        new_params[param_index] = np.clip(new_params[param_index],
                                          param_ranges[param_index][0],
                                          param_ranges[param_index][1])

        if new_params[2] >= new_params[3]:
            new_params[3] = new_params[2] + 1 + random.uniform(0, 0.5)
        if new_params[3] >= new_params[4]:
            new_params[4] = new_params[3] + 1 + random.uniform(0, 0.5)

        if new_params[5] <= new_params[2]:
            new_params[5] = new_params[2] + 0.1 + random.uniform(0, 0.5)
        if new_params[6] <= new_params[3]:
            new_params[6] = new_params[3] + 0.1 + random.uniform(0, 0.5)
        if new_params[7] <= new_params[4]:
            new_params[7] = new_params[4] + 0.1 + random.uniform(0, 0.5)

        new_score = objective_function(new_params)

        if new_score > current_score:
            current_params = new_params
            current_score = new_score
            no_improve_count = 0

            if new_score > best_score:
                best_params = new_params.copy()
                best_score = new_score
                history_best.append((best_params.copy(), best_score))

        else:
            if best_score > 0:
                quality_factor = min(1.0, current_score / best_score)
            else:
                quality_factor = 1.0

            denominator = temp * quality_factor
            if abs(denominator) < 1e-10:
                prob = 0.0
            else:
                prob = math.exp((new_score - current_score) / denominator)

            if random.random() < prob:
                current_params = new_params
                current_score = new_score

            no_improve_count += 1

        if no_improve_count >= restart_threshold:
            idx = random.randint(0, len(history_best) - 1)
            current_params, current_score = history_best[idx]

            for i in range(len(current_params)):
                if random.random() < 0.3:
                    if i == 0:
                        current_params[i] += random.uniform(-0.05, 0.05)
                    elif i == 1:
                        current_params[i] += random.uniform(-2, 2)
                    else:
                        current_params[i] += random.uniform(-0.2, 0.2)

            current_score = objective_function(current_params)
            no_improve_count = 0
            temp = max(temp * 1.5, final_temp * 2)

        progress = iteration / max_iterations
        if progress < 0.3:
            temp *= (cooling_rate ** 0.5)
        elif progress < 0.7:
            temp *= cooling_rate
        else:
            temp *= (cooling_rate ** 2)

        if temp < final_temp:
            temp = final_temp

        best_scores.append(best_score)
        iteration += 1

    return best_params, best_score


if __name__ == "__main__":
    best_params, best_score = adaptive_simulated_annealing(max_iterations=5000)
    print("最长有效遮蔽时间：",best_score)