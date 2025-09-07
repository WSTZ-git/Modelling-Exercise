#  问题四灵敏度分析可视化

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import time
plt.rc("font",family='DengXian')
plt.rcParams['axes.unicode_minus'] = False
g = 9.8
R_cloud = 10
v_sink = 3
cloud_lifetime = 20
cylinder_radius = 7
cylinder_height = 10
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
key_points[1, :] = T_base + np.array([0, 0, cylinder_height])

for i in range(n_points):
    angle = 2 * np.pi * i / n_points
    key_points[2 + i, :] = T_base + cylinder_radius * np.array([np.cos(angle), np.sin(angle), 0])

for i in range(n_points):
    angle = 2 * np.pi * i / n_points
    key_points[2 + n_points + i, :] = T_base + cylinder_radius * np.array([np.cos(angle), np.sin(angle), 0]) + np.array(
        [0, 0, cylinder_height])

def is_line_segment_intersect_sphere(p1, p2, center, radius):
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

def calculate_multiple_UAVs_shielding_time(theta, v_M1_current):
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
    is_effective = np.zeros(n_t, dtype=bool)
    has_cloud = np.zeros((n_t, 3), dtype=bool)

    for i in range(n_t):
        t = t_range[i]
        r_M1 = r_M1_0 + v_M1_current * t * e_M1

        any_cloud_exists = False
        r_C = np.zeros((3, 3))

        if t >= t_explosion1 and t <= t_explosion1 + cloud_lifetime:
            has_cloud[i, 0] = True
            any_cloud_exists = True
            r_C[0, :] = r_S1_explosion - v_sink * (t - t_explosion1) * np.array([0, 0, 1])

        if t >= t_explosion2 and t <= t_explosion2 + cloud_lifetime:
            has_cloud[i, 1] = True
            any_cloud_exists = True
            r_C[1, :] = r_S2_explosion - v_sink * (t - t_explosion2) * np.array([0, 0, 1])

        if t >= t_explosion3 and t <= t_explosion3 + cloud_lifetime:
            has_cloud[i, 2] = True
            any_cloud_exists = True
            r_C[2, :] = r_S3_explosion - v_sink * (t - t_explosion3) * np.array([0, 0, 1])

        if any_cloud_exists:
            is_effective_current = True

            for j in range(key_points.shape[0]):
                any_intersection = False

                for k in range(3):
                    if has_cloud[i, k]:
                        cloud_center = r_C[k, :]
                        if is_line_segment_intersect_sphere(r_M1, key_points[j, :], cloud_center, R_cloud):
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

def fitness_function(theta, v_M1_current):
    return -calculate_multiple_UAVs_shielding_time(theta, v_M1_current)

def run_optimization_for_velocity(v_M1_current):
    lb = [0.05, 80, 0.5, 0.3, 4, 70, 10, 5, 1.6, 80, 20, 5]
    ub = [0.15, 120, 1, 0.6, 4.5, 90, 15, 9, 2.2, 120, 30, 8]
    bounds = list(zip(lb, ub))

    options = {
        'maxiter': 30,
        'popsize': 15,
        'tol': 1e-6,
        'mutation': (0.5, 1),
        'recombination': 0.7,
        'disp': False,
        'polish': True
    }

    def fitness_wrapper(theta):
        return fitness_function(theta, v_M1_current)

    result = differential_evolution(fitness_wrapper, bounds, **options)
    optimal_params = result.x
    fval = result.fun

    total_effective_time = -fval

    return total_effective_time, optimal_params

def sensitivity_analysis():

    velocities = np.arange(200, 401, 10)

    effective_times = []
    optimal_parameters = []

    start_time = time.time()

    for i, v_missile in enumerate(velocities):

        try:
            eff_time, opt_params = run_optimization_for_velocity(v_missile)
            effective_times.append(eff_time)
            optimal_parameters.append(opt_params)

        except Exception as e:
            effective_times.append(0)
            optimal_parameters.append(None)

    total_time = time.time() - start_time

    return velocities, effective_times, optimal_parameters

def plot_sensitivity_results(velocities, effective_times):

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(velocities, effective_times, 'b-o', linewidth=2, markersize=6)
    plt.xlabel('导弹速度 (m/s)', fontsize=12)
    plt.ylabel('总有效遮蔽时间 (秒)', fontsize=12)
    plt.title('导弹速度对烟幕遮蔽效果的影响', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)

    for i, (v, t) in enumerate(zip(velocities, effective_times)):
        if i % 2 == 0:
            plt.annotate(f'{t:.2f}', (v, t), textcoords="offset points",
                         xytext=(0, 10), ha='center', fontsize=8)

    max_time_idx = np.argmax(effective_times)
    best_velocity = velocities[max_time_idx]
    best_time = effective_times[max_time_idx]

    plt.plot(best_velocity, best_time, 'ro', markersize=10,
             label=f'最佳点: v={best_velocity} m/s, t={best_time:.3f}s')
    plt.legend()

    plt.subplot(2, 1, 2)
    velocity_changes = np.diff(effective_times) / np.diff(velocities)
    velocity_mid = (velocities[1:] + velocities[:-1]) / 2

    plt.plot(velocity_mid, velocity_changes, 'r-s', linewidth=2, markersize=4)
    plt.xlabel('导弹速度 (m/s)', fontsize=12)
    plt.ylabel('遮蔽时间变化率 (秒/(m/s))', fontsize=12)
    plt.title('遮蔽效果对速度的敏感性', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    velocities, effective_times, optimal_parameters = sensitivity_analysis()
    plot_sensitivity_results(velocities, effective_times)