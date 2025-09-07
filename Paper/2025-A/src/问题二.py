# 问题二

import numpy as np

M1_0_position = np.array([20000.0, 0.0, 2000.0])
FY1_0_position = np.array([17800.0, 0.0, 1800.0])
v_missile = 300  # m/s
fake_target = np.array([0.0, 0.0, 0.0])  # 原点(假目标)
true_target = np.array([0.0, 200.0, 0.0])  # 真目标
effective_radius = 7  # m
sinking_speed = 3  # m/s
g = 9.8  # 重力加速度m/s
r0 = 10
r1 = 7
h0 = 10
y0 = 200

param_ranges = {
    'alpha': (3.13, 3.15),
    'v': (105, 110),
    't1': (0.5, 1.0),
    't2': (4.15, 4.45)
}


def get_valid_point(angle, z):
    x1 = r1 * np.cos(angle)
    y1 = y0 + r1 * np.sin(angle)
    z1 = z
    return [x1, y1, z1]


def obj(t, target, FY11_t_position):
    M1_sum = np.sqrt(M1_0_position[0] **2 + M1_0_position[1]** 2 + M1_0_position[2] ** 2)
    unit_dir = -M1_0_position / M1_sum
    M1_t_position = M1_0_position + v_missile * t * unit_dir
    OM1_length = np.sqrt(M1_t_position[0] **2 + M1_t_position[1]** 2 + M1_t_position[2] ** 2)

    x1, y1, z1 = target
    x_M1_t, y_M1_t, z_M1_t = M1_t_position
    x_FY11_t, y_FY11_t, z_FY11_t = FY11_t_position

    term1 = (x_M1_t - x1) * (x1 - x_FY11_t)
    term2 = (y_M1_t - y1) * (y1 - y_FY11_t)
    term3 = (z_M1_t - z1) * (z1 - z_FY11_t)
    dot_product = term1 + term2 + term3
    part1 = 4 * (dot_product **2)

    a_sq = (x_M1_t - x1)** 2 + (y_M1_t - y1) **2 + (z_M1_t - z1)** 2
    b_sq = (x1 - x_FY11_t) **2 + (y1 - y_FY11_t)** 2 + (z1 - z_FY11_t) ** 2
    part2 = 4 * a_sq * (b_sq - r0 **2)

    delta = part1 - part2

    return delta, a_sq, 2 * dot_product, x_M1_t, x1, y_M1_t, y1, z_M1_t, z1, OM1_length, M1_t_position


def valid(delta, a, b, xm, x1, ym, y1, zm, z1):
    if delta < 0:
        return False

    k1 = (-b + np.sqrt(delta)) / (2 * a)
    k2 = (-b - np.sqrt(delta)) / (2 * a)
    d1 = np.sqrt(pow(k1 * (xm - x1), 2) + pow(k1 * (ym - y1), 2) + pow(k1 * (zm - z1), 2))
    d2 = np.sqrt(pow(k2 * (xm - x1), 2) + pow(k2 * (ym - y1), 2) + pow(k2 * (zm - z1), 2))
    OM = np.sqrt((xm - x1) **2 + (ym - y1)** 2 + (zm - z1) ** 2)
    return d1 <= OM or d2 <= OM


def calculate_effective_time(params, return_details=False):
    alpha, v, t1, t2 = params

    if t2 <= t1:
        if return_details:
            return 0, [], None, None, None, None
        return 0, []

    flag = -1
    valid_times = []
    delta_records = []
    time_step = 1e-5
    max_time = min(t2 + 20, 50)

    earliest_time = None
    latest_time = None
    M1_earliest_pos = None
    FY11_earliest_pos = None
    M1_latest_pos = None
    FY11_latest_pos = None

    for t in np.arange(t2 + 0.1, max_time, time_step):
        FY1_t1_position = np.array([
            FY1_0_position[0] + v * t1 * np.cos(alpha),
            FY1_0_position[1] + v * t1 * np.sin(alpha),
            FY1_0_position[2]
        ])

        FY11_t2_position = np.array([
            FY1_t1_position[0] + v * (t2 - t1) * np.cos(alpha),
            FY1_t1_position[1] + v * (t2 - t1) * np.sin(alpha),
            FY1_t1_position[2] - g * (t2 - t1) **2 / 2
        ])

        FY11_t_position = np.array([
            FY11_t2_position[0],
            FY11_t2_position[1],
            FY11_t2_position[2] - sinking_speed * (t - t2)
        ])

        angles = np.arange(0, 2 * np.pi, np.pi / 8)
        heights = np.arange(0, h0, h0 / 5)
        is_valid = False
        max_delta = -np.inf

        for angle in angles:
            for z in heights:
                target = get_valid_point(angle, z)
                result = obj(t, target, FY11_t_position)
                delta, a_sq, b_val, xm, x1, ym, y1, zm, z1, OM1_length, M1_t_position = result

                if delta > max_delta:
                    max_delta = delta

                if delta >= 0:
                    if valid(delta, a_sq, b_val, xm, x1, ym, y1, zm, z1):
                        is_valid = True
                        break
            if is_valid:
                break

        delta_records.append((t, max_delta))

        if is_valid:
            if flag < 0:
                flag = flag * -1
                earliest_time = t
                M1_earliest_pos = M1_t_position.copy()
                FY11_earliest_pos = FY11_t_position.copy()
            valid_times.append(t)
            latest_time = t
            M1_latest_pos = M1_t_position.copy()
            FY11_latest_pos = FY11_t_position.copy()
        else:
            if flag > 0:
                flag = flag * -1
                break

    try:
        if len(valid_times) < 2:
            if return_details:
                return 0, delta_records, None, None, None, None
            return 0, delta_records

        time_window = max(valid_times) - min(valid_times)

        if return_details:
            return (time_window, delta_records,
                    earliest_time, latest_time,
                    (M1_earliest_pos, FY11_earliest_pos),
                    (M1_latest_pos, FY11_latest_pos))
        return time_window, delta_records
    except:
        if return_details:
            return 0, delta_records, None, None, None, None
        return 0, delta_records

def objective_function(params):
    time_window, _ = calculate_effective_time(params)
    return -time_window

def simulated_annealing(initial_params, max_iter=500, initial_temp=100, final_temp=0.01,cooling_rate=0.95):
    current_params = np.array(initial_params)
    current_score, _ = calculate_effective_time(current_params)
    best_params = current_params.copy()
    best_score = current_score

    best_details = None

    scores = [current_score]
    temp = initial_temp
    count = 0
    patience = 50

    while temp>final_temp:
        for i in range(max_iter):
            scale = 0.1 + 0.9 * (temp / initial_temp)

            new_params = current_params.copy()
            new_params[0] += np.random.normal(0, 0.1 * scale)
            new_params[1] += np.random.normal(0, 2 * scale)
            new_params[2] += np.random.normal(0, 0.2 * scale)

            new_params[3] += np.random.normal(0, 0.2 * scale)
            if new_params[3] <= new_params[2]:
                new_params[3] = new_params[2] + 0.1 + np.random.random() * 0.5 * scale

            new_params[0] = np.clip(new_params[0], *param_ranges['alpha'])
            new_params[1] = np.clip(new_params[1], *param_ranges['v'])
            new_params[2] = np.clip(new_params[2], *param_ranges['t1'])
            new_params[3] = np.clip(new_params[3], *param_ranges['t2'])

            new_score, _ = calculate_effective_time(new_params)

            score_diff = new_score - current_score

            if score_diff > 0 or np.random.random() < np.exp(score_diff / temp):
                current_params = new_params
                current_score = new_score

                if current_score > best_score:
                    best_params = current_params.copy()
                    best_score = current_score
                    count = 0

                    _, _, earliest, latest, earliest_pos, latest_pos = calculate_effective_time(
                        best_params, return_details=True)
                    best_details = (earliest, latest, earliest_pos, latest_pos)
                else:
                    count += 1
            else:
                count += 1

            if count >= patience:
                break
            scores.append(current_score)
        temp *= cooling_rate

    return best_params, best_score, scores, best_details

initial_alpha = 3.13
initial_v = 109.78
initial_t1 = 0.71
initial_t2 = 4.23
initial_params = [initial_alpha, initial_v, initial_t1, initial_t2]

(initial_time_window, initial_delta_data,
 initial_earliest, initial_latest,
 initial_earliest_pos, initial_latest_pos) = calculate_effective_time(initial_params, return_details=True)

if initial_earliest and initial_latest:
    M1_earliest_init, FY11_earliest_init = initial_earliest_pos
    M1_latest_init, FY11_latest_init = initial_latest_pos

best_params, best_score, scores, best_details = simulated_annealing(
    initial_params,
    max_iter=500,
    initial_temp=10,
    final_temp=0.01,
    cooling_rate=0.95
)


initial_alpha = 3.13
initial_v = 109.78
initial_t1 = 0.71
initial_t2 = 4.23
initial_params = [initial_alpha, initial_v, initial_t1, initial_t2]

(initial_time_window, initial_delta_data,
 initial_earliest, initial_latest,
 initial_earliest_pos, initial_latest_pos) = calculate_effective_time(initial_params, return_details=True)

print(f"最优有效遮蔽时间: {initial_time_window:.6f}秒")