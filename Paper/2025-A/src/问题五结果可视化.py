#  问题五结果可视化

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

plt.rc("font", family='DengXian')
plt.rcParams['axes.unicode_minus'] = False

def create_comprehensive_visualization( r_missiles_0,r_UAVs_0,
                                       key_points,  UAV_contributions,
                                       missile_contributions, release_pos_all,
                                       explosion_pos_all):
    fig = plt.figure(figsize=(16, 7))

    fig.subplots_adjust(
        left=0.08,
        right=0.95,
        top=0.90,
        bottom=0.15,
        wspace=0.35,
        hspace=0.3
    )

    ax2 = fig.add_subplot(1, 2, 1)
    plot_2d_overview(ax2, r_missiles_0, r_UAVs_0, key_points, release_pos_all, explosion_pos_all)
    ax4 = fig.add_subplot(1, 2, 2)
    plot_uav_contributions(ax4, UAV_contributions, missile_contributions)
    fig.suptitle('导弹防御系统分析报告', fontsize=16, fontweight='bold', y=0.95)
    plt.show()

def plot_2d_overview(ax, r_missiles_0, r_UAVs_0, key_points, release_pos_all, explosion_pos_all):
    """绘制2D俯视图"""

    colors_missiles = ['red', 'orange', 'darkred']
    for i in range(3):
        ax.scatter(r_missiles_0[i, 0], r_missiles_0[i, 1],
                   color=colors_missiles[i], s=120, marker='^', label=f'导弹M{i + 1}')

    colors_uav = ['blue', 'green', 'purple', 'cyan', 'magenta']
    for i in range(5):
        ax.scatter(r_UAVs_0[i, 0], r_UAVs_0[i, 1],
                   color=colors_uav[i], s=100, marker='o', label=f'无人机FY{i + 1}')

        for j in range(3):
            ax.plot([r_UAVs_0[i, 0], release_pos_all[i][j, 0], explosion_pos_all[i][j, 0]],
                    [r_UAVs_0[i, 1], release_pos_all[i][j, 1], explosion_pos_all[i][j, 1]],
                    color=colors_uav[i], alpha=0.5, linewidth=1)
            ax.scatter(explosion_pos_all[i][j, 0], explosion_pos_all[i][j, 1],
                       color=colors_uav[i], s=40, marker='*', alpha=0.7)

    target_circle = Circle((key_points[0, 0], key_points[0, 1]), 7,
                           fill=False, color='black', linewidth=2)
    ax.add_patch(target_circle)
    ax.scatter(key_points[0, 0], key_points[0, 1], color='black', s=150, marker='h', label='真实目标')

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('俯视图 - 战场部署', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

def plot_uav_contributions(ax, UAV_contributions, missile_contributions):
    """绘制无人机贡献条形图"""

    uav_names = [f'FY{i + 1}' for i in range(5)]
    x = np.arange(len(uav_names))
    width = 0.2

    colors = ['red', 'orange', 'darkred']
    for i in range(3):
        ax.bar(x + i * width, missile_contributions[:, i], width,
               label=f'对导弹M{i + 1}', color=colors[i], alpha=0.8)

    ax.set_xlabel('无人机')
    ax.set_ylabel('遮蔽时间 (s)')
    ax.set_title('各无人机对导弹的遮蔽贡献', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(uav_names)
    ax.legend()
    ax.grid(True, alpha=0.3)

def main_combined_evaluation():
    g = 9.8
    v_missile = 300
    R_cloud = 10
    v_sink = 3
    cloud_lifetime = 20
    cylinder_radius = 7
    cylinder_height = 10
    min_interval = 1
    max_flares_per_uav = 3

    r_M1_0 = np.array([20000, 0, 2000])
    r_M2_0 = np.array([19000, 600, 2100])
    r_M3_0 = np.array([18000, -600, 1900])
    r_missiles_0 = np.vstack([r_M1_0, r_M2_0, r_M3_0])

    O = np.array([0, 0, 0])
    T_base = np.array([0, 200, 0])

    r_FY1_0 = np.array([17800, 0, 1800])
    r_FY2_0 = np.array([12000, 1400, 1400])
    r_FY3_0 = np.array([6000, -3000, 700])
    r_FY4_0 = np.array([11000, 2000, 1800])
    r_FY5_0 = np.array([13000, -2000, 1300])
    r_UAVs_0 = np.vstack([r_FY1_0, r_FY2_0, r_FY3_0, r_FY4_0, r_FY5_0])
    e_M1 = (O - r_M1_0) / np.linalg.norm(O - r_M1_0)
    e_M2 = (O - r_M2_0) / np.linalg.norm(O - r_M2_0)
    e_M3 = (O - r_M3_0) / np.linalg.norm(O - r_M3_0)
    e_missiles = np.vstack([e_M1, e_M2, e_M3])
    t_start = 0
    t_end = 100
    dt = 0.01
    t_range = np.arange(t_start, t_end + dt, dt)
    n_t = len(t_range)

    n_points = 2
    key_points = np.zeros((2 * n_points + 2, 3))
    key_points[0, :] = T_base
    key_points[1, :] = T_base + np.array([0, 0, cylinder_height])
    for i in range(1, n_points + 1):
        angle = 2 * np.pi * (i - 1) / n_points
        key_points[2 + i - 1, :] = T_base + cylinder_radius * np.array([np.cos(angle), np.sin(angle), 0])

    for i in range(1, n_points + 1):
        angle = 2 * np.pi * (i - 1) / n_points
        key_points[2 + n_points + i - 1, :] = T_base + cylinder_radius * np.array(
            [np.cos(angle), np.sin(angle), 0]) + np.array([0, 0, cylinder_height])

    dist_to_target = [np.linalg.norm(r_M1_0 - T_base),
                      np.linalg.norm(r_M2_0 - T_base),
                      np.linalg.norm(r_M3_0 - T_base)]
    est_flight_time = np.array(dist_to_target) / v_missile

    UAV_params = [
        {
            'direction_angle': 0.13,
            'speed': 72.00,
            'release_time': np.array([0.01, 1.00, 18.51]),
            'delay_time': np.array([0.01, 0.45, 0.83]),
            'explosion_time': np.array([0.02, 1.00, 18.51])
        },
        {
            'direction_angle': 5.15,
            'speed': 127.02,
            'release_time': np.array([7.9, 13.14, 25.33]),
            'delay_time': np.array([0.6, 2.68, 7.32]),
            'explosion_time': np.array([8.5, 13.75, 25.94])
        },
        {
            'direction_angle': 1.39,
            'speed': 82.34,
            'release_time': np.array([33.9, 34.945, 35.91]),
            'delay_time': np.array([0.51, 2.5845, 2.23]),
            'explosion_time': np.array([34.5, 35.429, 36.43])
        },
        {
            'direction_angle': 0.00,
            'speed': 70.00,
            'release_time': np.array([0.92, 1.92, 2.92]),
            'delay_time': np.array([0.50, 1.34, 0.50]),
            'explosion_time': np.array([1.42, 2.42, 3.42])
        },
        {
            'direction_angle': 2.10,
            'speed': 104.78,
            'release_time': np.array([16.04, 17.59, 21.57]),
            'delay_time': np.array([1.65, 4.69, 1.89]),
            'explosion_time': np.array([17.69, 19.24, 23.22])
        }
    ]
    for i in range(len(UAV_params)):
        UAV_params[i]['direction_angle_deg'] = np.mod(UAV_params[i]['direction_angle'] * 180 / np.pi, 360)

    num_UAVs = 5
    UAV_contributions = np.zeros(num_UAVs)
    missile_contributions = np.zeros((num_UAVs, 3))
    release_pos_all = [np.zeros((3, 3)) for _ in range(num_UAVs)]
    explosion_pos_all = [np.zeros((3, 3)) for _ in range(num_UAVs)]
    explosion_time_all = [np.zeros(3) for _ in range(num_UAVs)]

    for uav_idx in range(num_UAVs):
        UAV_param = UAV_params[uav_idx]
        e_UAV = np.array([np.cos(UAV_param['direction_angle']),
                          np.sin(UAV_param['direction_angle']), 0])

        release_pos = np.zeros((3, 3))
        explosion_pos = np.zeros((3, 3))
        explosion_time = np.zeros(3)

        for j in range(3):
            release_pos[j, :] = r_UAVs_0[uav_idx, :] + UAV_param['speed'] * UAV_param['release_time'][j] * e_UAV
            explosion_time[j] = UAV_param['explosion_time'][j]
            explosion_pos[j, :] = (release_pos[j, :] +
                                   UAV_param['speed'] * UAV_param['delay_time'][j] * e_UAV -
                                   0.5 * g * (UAV_param['delay_time'][j] ** 2) * np.array([0, 0, 1]))

        release_pos_all[uav_idx] = release_pos
        explosion_pos_all[uav_idx] = explosion_pos
        explosion_time_all[uav_idx] = explosion_time

    for uav_idx in range(num_UAVs):
        temp_UAV_params = [{} for _ in range(num_UAVs)]
        for i in range(num_UAVs):
            if i == uav_idx:
                temp_UAV_params[i] = UAV_params[i].copy()
            else:
                temp_UAV_params[i] = {
                    'direction_angle': 0,
                    'direction_angle_deg': 0,
                    'speed': 100,
                    'release_time': np.array([1000, 1000, 1000]),
                    'delay_time': np.array([1, 1, 1]),
                    'explosion_time': np.array([1001, 1001, 1001])
                }

        total_contribution, missile_times, _ = evaluate_combined_solution(
            temp_UAV_params, r_missiles_0, e_missiles, v_missile,
            r_UAVs_0, g, R_cloud, v_sink, cloud_lifetime,
            key_points, t_range, dt
        )
        UAV_contributions[uav_idx] = total_contribution
        missile_contributions[uav_idx, :] = missile_times

    total_time, missile_times, effective_intervals = evaluate_combined_solution(
        UAV_params, r_missiles_0, e_missiles, v_missile,
        r_UAVs_0, g, R_cloud, v_sink, cloud_lifetime,
        key_points, t_range, dt
    )

    for uav_idx in range(num_UAVs):
        UAV_param = UAV_params[uav_idx]
        for j in range(3):
            flare_params = (
                explosion_pos_all[uav_idx][j, :],
                explosion_time_all[uav_idx][j],
            )
            flare_times = evaluate_single_flare_contribution(
                flare_params, r_missiles_0, e_missiles, v_missile,
                g, R_cloud, v_sink, cloud_lifetime,
                key_points, t_range, dt
            )
    create_comprehensive_visualization(
         r_missiles_0,
        r_UAVs_0,
        key_points, UAV_contributions,
        missile_contributions, release_pos_all,
        explosion_pos_all
    )

    return UAV_params, total_time, missile_times, effective_intervals

def isLineSegmentIntersectSphere(p1, p2, center, radius):
    """判断线段与球体是否相交（修正向量维度）"""
    p1 = p1.flatten()
    p2 = p2.flatten()
    center = center.flatten()

    dist1 = np.linalg.norm(p1 - center)
    dist2 = np.linalg.norm(p2 - center)
    if dist1 <= radius or dist2 <= radius:
        return True

    v = p2 - p1
    a = np.dot(v, v)
    b = 2 * np.dot(p1 - center, v)
    c = np.dot(p1 - center, p1 - center) - radius ** 2
    discriminant = b ** 2 - 4 * a * c

    if discriminant < 0:
        return False

    t1 = (-b + np.sqrt(discriminant)) / (2 * a)
    t2 = (-b - np.sqrt(discriminant)) / (2 * a)
    if (0 <= t1 <= 1) or (0 <= t2 <= 1):
        return True
    else:
        return False


def evaluate_combined_solution_with_status(UAV_params, r_missiles_0, e_missiles, v_missile,
                                           r_UAVs_0, g, R_cloud, v_sink, cloud_lifetime,
                                           key_points, t_range, dt):
    """带遮蔽状态的合并效果评估（原文档逻辑）"""
    num_UAVs = len(UAV_params)
    n_t = len(t_range)
    num_missiles = 3

    e_UAVs = np.zeros((num_UAVs, 3))
    for i in range(num_UAVs):
        e_UAVs[i, :] = np.array([np.cos(UAV_params[i]['direction_angle']),
                                 np.sin(UAV_params[i]['direction_angle']), 0])

    release_pos = [np.zeros((3, 3)) for _ in range(num_UAVs)]
    explosion_pos = [np.zeros((3, 3)) for _ in range(num_UAVs)]
    for i in range(num_UAVs):
        for j in range(3):
            release_pos[i][j, :] = (r_UAVs_0[i, :] +
                                    UAV_params[i]['speed'] * UAV_params[i]['release_time'][j] * e_UAVs[i, :])
            explosion_pos[i][j, :] = (release_pos[i][j, :] +
                                      UAV_params[i]['speed'] * UAV_params[i]['delay_time'][j] * e_UAVs[i, :] -
                                      0.5 * g * (UAV_params[i]['delay_time'][j] ** 2) * np.array([0, 0, 1]))

    r_missiles = np.zeros((n_t, num_missiles, 3))
    r_clouds = [[np.zeros((n_t, 3)) for _ in range(3)] for __ in range(num_UAVs)]
    is_effective = np.zeros((n_t, num_missiles))
    has_cloud = np.zeros((n_t, num_UAVs, 3))

    for i in range(n_t):
        t = t_range[i]
        for j in range(num_missiles):
            r_missiles[i, j, :] = r_missiles_0[j, :] + v_missile * t * e_missiles[j, :]

        for j in range(num_UAVs):
            for k in range(3):
                if (t >= UAV_params[j]['explosion_time'][k]) and (
                        t <= UAV_params[j]['explosion_time'][k] + cloud_lifetime):
                    has_cloud[i, j, k] = 1
                    cloud_center = (explosion_pos[j][k, :] -
                                    v_sink * (t - UAV_params[j]['explosion_time'][k]) * np.array([0, 0, 1]))
                    r_clouds[j][k][i, :] = cloud_center

        for j in range(num_missiles):
            all_points_covered = True
            for k in range(key_points.shape[0]):
                point_covered = False
                for m in range(num_UAVs):
                    for n in range(3):
                        if has_cloud[i, m, n] == 1:
                            cloud_center = r_clouds[m][n][i, :]
                            if isLineSegmentIntersectSphere(
                                    r_missiles[i, j, :].reshape(-1, 1),
                                    key_points[k, :].reshape(-1, 1),
                                    cloud_center, R_cloud
                            ):
                                point_covered = True
                                break
                    if point_covered:
                        break
                if not point_covered:
                    all_points_covered = False
                    break
            is_effective[i, j] = all_points_covered

    effective_intervals = [np.array([]) for _ in range(num_missiles)]
    for j in range(num_missiles):
        effective_idx = np.where(is_effective[:, j] == 1)[0]
        if len(effective_idx) == 0:
            continue

        intervals = []
        interval_start = t_range[effective_idx[0]]
        prev_idx = effective_idx[0]
        for k in range(1, len(effective_idx)):
            if effective_idx[k] - prev_idx > 1:
                interval_end = t_range[prev_idx]
                intervals.append([interval_start, interval_end])
                interval_start = t_range[effective_idx[k]]
            prev_idx = effective_idx[k]
        interval_end = t_range[effective_idx[-1]]
        intervals.append([interval_start, interval_end])
        effective_intervals[j] = np.array(intervals)

    missile_times = np.zeros(num_missiles)
    for j in range(num_missiles):
        if effective_intervals[j].size > 0:
            missile_times[j] = np.sum(effective_intervals[j][:, 1] - effective_intervals[j][:, 0])
    total_time = np.sum(missile_times)

    return total_time, missile_times, effective_intervals, is_effective

def evaluate_combined_solution(UAV_params, r_missiles_0, e_missiles, v_missile,
                               r_UAVs_0, g, R_cloud, v_sink, cloud_lifetime,
                               key_points, t_range, dt):
    """合并效果评估（调用带状态评估，屏蔽状态输出）"""
    total_time, missile_times, effective_intervals, _ = evaluate_combined_solution_with_status(
        UAV_params, r_missiles_0, e_missiles, v_missile,
        r_UAVs_0, g, R_cloud, v_sink, cloud_lifetime,
        key_points, t_range, dt
    )
    return total_time, missile_times, effective_intervals


def evaluate_single_flare_contribution(flare_params, r_missiles_0, e_missiles, v_missile,
                                       g, R_cloud, v_sink, cloud_lifetime,
                                       key_points, t_range, dt):
    num_missiles = 3
    n_t = len(t_range)
    flare_explosion_pos, flare_explosion_time = flare_params

    is_effective = np.zeros((n_t, num_missiles))

    for i in range(n_t):
        t = t_range[i]
        missile_positions = np.zeros((num_missiles, 3))
        for j in range(num_missiles):
            missile_positions[j, :] = r_missiles_0[j, :] + v_missile * t * e_missiles[j, :]

        if (t >= flare_explosion_time) and (t <= flare_explosion_time + cloud_lifetime):
            cloud_center = flare_explosion_pos - v_sink * (t - flare_explosion_time) * np.array([0, 0, 1])
            for j in range(num_missiles):
                all_points_covered = True
                for k in range(key_points.shape[0]):
                    if not isLineSegmentIntersectSphere(
                            missile_positions[j, :].reshape(-1, 1),
                            key_points[k, :].reshape(-1, 1),
                            cloud_center, R_cloud
                    ):
                        all_points_covered = False
                        break
                is_effective[i, j] = all_points_covered
        else:
            is_effective[i, :] = 0

    missile_times = np.zeros(num_missiles)
    for j in range(num_missiles):
        effective_idx = np.where(is_effective[:, j] == 1)[0]
        if len(effective_idx) == 0:
            continue
        intervals = []
        interval_start = t_range[effective_idx[0]]
        prev_idx = effective_idx[0]
        for k in range(1, len(effective_idx)):
            if effective_idx[k] - prev_idx > 1:
                interval_end = t_range[prev_idx]
                intervals.append([interval_start, interval_end])
                interval_start = t_range[effective_idx[k]]
            prev_idx = effective_idx[k]
        interval_end = t_range[effective_idx[-1]]
        intervals.append([interval_start, interval_end])
        missile_times[j] = np.sum(np.array(intervals)[:, 1] - np.array(intervals)[:, 0])
    return missile_times

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results_{timestamp}.txt"

    import sys


    class Tee:
        def __init__(self, file_name):
            self.file = open(file_name, 'w')
            self.stdout = sys.stdout
            sys.stdout = self

        def write(self, data):
            self.file.write(data)
            self.stdout.write(data)

        def flush(self):
            self.file.flush()
            self.stdout.flush()

        def close(self):
            sys.stdout = self.stdout
            self.file.close()

    tee = Tee(filename)

    try:
        main_combined_evaluation()
    except Exception as e:
        import traceback

        traceback.print_exc()

    tee.close()