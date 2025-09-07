#  问题四

import numpy as np
from scipy.optimize import differential_evolution

g = 9.8
v_M1 = 300
R_cloud = 10
v_sink = 3
cloud_lifetime = 20
r0 = 7
h0 = 10

r_M1_0 = np.array([20000, 0, 2000])
O = np.array([0, 0, 0])
T_base = np.array([0, 200, 0])

r_FY1_0 = np.array([17800, 0, 1800])
r_FY2_0 = np.array([12000, 1400, 1400])
r_FY3_0 = np.array([6000, -3000, 700])

e_M1 = (O - r_M1_0) / np.linalg.norm(O - r_M1_0)

t_start = 0
t_end = 100
dt = 0.05
t_range = np.arange(t_start, t_end + dt, dt)
n_t = len(t_range)

n_points = 2
key_points = np.zeros((2 * n_points + 2, 3))

key_points[0, :] = T_base
key_points[1, :] = T_base + np.array([0, 0, h0])

for i in range(n_points):
    angle = 2 * np.pi * i / n_points
    key_points[2 + i, :] = T_base + r0 * np.array([np.cos(angle), np.sin(angle), 0])

for i in range(n_points):
    angle = 2 * np.pi * i / n_points
    key_points[2 + n_points + i, :] = T_base + r0 * np.array([np.cos(angle), np.sin(angle), 0]) + np.array(
        [0, 0, h0])

dist_to_target = np.linalg.norm(r_M1_0 - T_base)
est_flight_time = dist_to_target / v_M1

def check_intersect(p1, p2, center, radius):
    d = p2 - p1
    a = p1 - center

    a_dot_d = np.dot(a, d)
    d_dot_d = np.dot(d, d)
    a_dot_a = np.dot(a, a)
    r_sq = radius ** 2

    discriminant = a_dot_d ** 2 - d_dot_d * (a_dot_a - r_sq)

    if discriminant < 0:
        return False

    sqrt_discriminant = np.sqrt(discriminant)

    t1 = (-a_dot_d + sqrt_discriminant) / d_dot_d
    t2 = (-a_dot_d - sqrt_discriminant) / d_dot_d

    if (0 <= t1 <= 1) or (0 <= t2 <= 1):
        return True
    return False

def cal_mul_time(theta, r_M1_0, e_M1, v_M1, r_FY1_0, r_FY2_0, r_FY3_0, g, R_cloud, v_sink,
                                           cloud_lifetime, key_points, t_range, dt):
    angle1, speed1, t_release1, t_delay1 = theta[0:4]
    angle2, speed2, t_release2, t_delay2 = theta[4:8]
    angle3, speed3, t_release3, t_delay3 = theta[8:12]

    t_explosion1 = t_release1 + t_delay1
    t_explosion2 = t_release2 + t_delay2
    t_explosion3 = t_release3 + t_delay3

    e_FY1 = np.array([np.cos(angle1), np.sin(angle1), 0])
    e_FY2 = np.array([np.cos(angle2), np.sin(angle2), 0])
    e_FY3 = np.array([np.cos(angle3), np.sin(angle3), 0])

    r_FY1_release = r_FY1_0 + speed1 * t_release1 * e_FY1
    r_S1_explosion = r_FY1_release + t_delay1 * speed1 * e_FY1 - 0.5 * g * t_delay1 ** 2 * np.array([0, 0, 1])

    r_FY2_release = r_FY2_0 + speed2 * t_release2 * e_FY2
    r_S2_explosion = r_FY2_release + t_delay2 * speed2 * e_FY2 - 0.5 * g * t_delay2 ** 2 * np.array([0, 0, 1])

    r_FY3_release = r_FY3_0 + speed3 * t_release3 * e_FY3
    r_S3_explosion = r_FY3_release + t_delay3 * speed3 * e_FY3 - 0.5 * g * t_delay3 ** 2 * np.array([0, 0, 1])

    n_t = len(t_range)

    r_M1 = np.zeros((n_t, 3))
    r_C = np.zeros((n_t, 3, 3))
    is_effective = np.zeros(n_t, dtype=bool)
    has_cloud = np.zeros((n_t, 3), dtype=bool)

    for i in range(n_t):
        t = t_range[i]
        r_M1[i, :] = r_M1_0 + v_M1 * t * e_M1

        any_cloud_exists = False

        if t >= t_explosion1 and t <= t_explosion1 + cloud_lifetime:
            has_cloud[i, 0] = True
            any_cloud_exists = True
            r_C[i, 0, :] = r_S1_explosion - v_sink * (t - t_explosion1) * np.array([0, 0, 1])

        if t >= t_explosion2 and t <= t_explosion2 + cloud_lifetime:
            has_cloud[i, 1] = True
            any_cloud_exists = True
            r_C[i, 1, :] = r_S2_explosion - v_sink * (t - t_explosion2) * np.array([0, 0, 1])

        if t >= t_explosion3 and t <= t_explosion3 + cloud_lifetime:
            has_cloud[i, 2] = True
            any_cloud_exists = True
            r_C[i, 2, :] = r_S3_explosion - v_sink * (t - t_explosion3) * np.array([0, 0, 1])

        if any_cloud_exists:
            is_effective_current = True

            for j in range(key_points.shape[0]):
                any_intersection = False

                for k in range(3):
                    if has_cloud[i, k]:
                        cloud_center = r_C[i, k, :]
                        if check_intersect(r_M1[i, :], key_points[j, :], cloud_center, R_cloud):
                            any_intersection = True
                            break

                if not any_intersection:
                    is_effective_current = False
                    break

            is_effective[i] = is_effective_current

    effective_idx = np.where(is_effective)[0]
    if len(effective_idx) == 0:
        return 0

    effective_intervals = []
    interval_start = t_range[effective_idx[0]]
    prev_idx = effective_idx[0]

    for j in range(1, len(effective_idx)):
        if effective_idx[j] - prev_idx > 1:
            interval_end = t_range[prev_idx]
            effective_intervals.append([interval_start, interval_end])
            interval_start = t_range[effective_idx[j]]
        prev_idx = effective_idx[j]

    interval_end = t_range[effective_idx[-1]]
    effective_intervals.append([interval_start, interval_end])
    effective_intervals = np.array(effective_intervals)

    effective_time = np.sum(effective_intervals[:, 1] - effective_intervals[:, 0])
    return effective_time

def fitness_function(theta):
    return -cal_mul_time(theta,r_M1_0, e_M1, v_M1, r_FY1_0, r_FY2_0, r_FY3_0, g, R_cloud, v_sink,
                        cloud_lifetime,key_points, t_range, dt)

lb = [0.05, 80, 0.5, 0.3, 4, 70, 10, 5, 1.6, 80, 20, 5]
ub = [0.15, 120, 1, 0.6, 4.5, 90, 15, 9, 2.2, 120, 30, 8]

bounds = list(zip(lb, ub))

options = {
    'maxiter': 50,
    'popsize': 20,
    'tol': 1e-6,
    'mutation': (0.5, 1),
    'recombination': 0.7,
    'disp': True,
    'polish': True
}

result = differential_evolution(fitness_function, bounds, **options)
optimal_params = result.x
fval = result.fun

angle1 = optimal_params[0]
angle1_deg = angle1 * 180 / np.pi
speed1 = optimal_params[1]
release_time1 = optimal_params[2]
delay_time1 = optimal_params[3]
explosion_time1 = release_time1 + delay_time1

angle2 = optimal_params[4]
angle2_deg = angle2 * 180 / np.pi
speed2 = optimal_params[5]
release_time2 = optimal_params[6]
delay_time2 = optimal_params[7]
explosion_time2 = release_time2 + delay_time2

angle3 = optimal_params[8]
angle3_deg = angle3 * 180 / np.pi
speed3 = optimal_params[9]
release_time3 = optimal_params[10]
delay_time3 = optimal_params[11]
explosion_time3 = release_time3 + delay_time3

e_FY1 = np.array([np.cos(angle1), np.sin(angle1), 0])
e_FY2 = np.array([np.cos(angle2), np.sin(angle2), 0])
e_FY3 = np.array([np.cos(angle3), np.sin(angle3), 0])

release_pos1 = r_FY1_0 + speed1 * release_time1 * e_FY1
explosion_pos1 = release_pos1 + delay_time1 * speed1 * e_FY1 - 0.5 * g * delay_time1 ** 2 * np.array([0, 0, 1])

release_pos2 = r_FY2_0 + speed2 * release_time2 * e_FY2
explosion_pos2 = release_pos2 + delay_time2 * speed2 * e_FY2 - 0.5 * g * delay_time2 ** 2 * np.array([0, 0, 1])

release_pos3 = r_FY3_0 + speed3 * release_time3 * e_FY3
explosion_pos3 = release_pos3 + delay_time3 * speed3 * e_FY3 - 0.5 * g * delay_time3 ** 2 * np.array([0, 0, 1])

total_effective_time = -fval
print(f'总有效遮蔽时间: {total_effective_time:.2f} 秒')