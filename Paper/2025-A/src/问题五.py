#  问题五

import numpy as np
from scipy.optimize import differential_evolution
import time
from datetime import datetime

def main_combined_evaluation():
    g = 9.8  
    v_missile = 300  
    R_cloud = 10  
    v_sink = 3  
    shadow_t = 20  
    cylinder_radius = 7  
    cylinder_height = 10  
    min_interval = 1  
    max_flares_per_K = 3  

    r_M1_0 = np.array([20000, 0, 2000]) 
    r_M2_0 = np.array([19000, 600, 2100]) 
    r_M3_0 = np.array([18000, -600, 1900]) 
    r_missile0 = np.vstack([r_M1_0, r_M2_0, r_M3_0]) 

    O = np.array([0, 0, 0]) 
    T_base = np.array([0, 200, 0])

    r_FY1_0 = np.array([17800, 0, 1800])
    r_FY2_0 = np.array([12000, 1400, 1400])
    r_FY3_0 = np.array([6000, -3000, 700])
    r_FY4_0 = np.array([11000, 2000, 1800])
    r_FY5_0 = np.array([13000, -2000, 1300])
    r_Ks0 = np.vstack([r_FY1_0, r_FY2_0, r_FY3_0, r_FY4_0, r_FY5_0])

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
    keypoints = np.zeros((2 * n_points + 2, 3))
    keypoints[0, :] = T_base
    keypoints[1, :] = T_base + np.array([0, 0, cylinder_height])
    for i in range(1, n_points + 1):
        angle = 2 * np.pi * (i - 1) / n_points
        keypoints[2 + i - 1, :] = T_base + cylinder_radius * np.array([np.cos(angle), np.sin(angle), 0])
        
    for i in range(1, n_points + 1):
        angle = 2 * np.pi * (i - 1) / n_points
        keypoints[2 + n_points + i - 1, :] = T_base + cylinder_radius * np.array(
            [np.cos(angle), np.sin(angle), 0]) + np.array([0, 0, cylinder_height])

    dist_to_target = [np.linalg.norm(r_M1_0 - T_base),
                      np.linalg.norm(r_M2_0 - T_base),
                      np.linalg.norm(r_M3_0 - T_base)]
    est_flight_time = np.array(dist_to_target) / v_missile

    K_params = [
        {
            'direction_angle': 0.13,
            'speed': 72.00,
            'release_time': np.array([0.00, 1.00, 18.51]),
            'delay_time': np.array([0.00, 0.45, 0.83]),
            'explosion_time': np.array([0.00, 1.00, 18.51])
        },
        # 无人机FY2
        {
            'direction_angle': 5.15,
            'speed': 127.02,
            'release_time': np.array([7.91, 13.14, 25.33]),
            'delay_time': np.array([0.61, 2.68, 7.32]),
            'explosion_time': np.array([8.52, 13.75, 25.94])
        },
        # 无人机FY3
        {
            'direction_angle': 1.39,
            'speed': 82.34,
            'release_time': np.array([33.91, 34.91, 35.91]),
            'delay_time': np.array([0.52, 2.58, 2.23]),
            'explosion_time': np.array([34.43, 35.43, 36.43])
        },
        # 无人机FY4
        {
            'direction_angle': 0.00,
            'speed': 70.00,
            'release_time': np.array([0.92, 1.92, 2.92]),
            'delay_time': np.array([0.50, 1.34, 0.50]),
            'explosion_time': np.array([1.42, 2.42, 3.42])
        },
        # 无人机FY5
        {
            'direction_angle': 2.10,
            'speed': 104.78,
            'release_time': np.array([16.04, 17.59, 21.57]),
            'delay_time': np.array([1.65, 4.69, 1.89]),
            'explosion_time': np.array([17.69, 19.24, 23.22])
        }
    ]
    
    for i in range(len(K_params)):
        K_params[i]['direction_angle_deg'] = np.mod(K_params[i]['direction_angle'] * 180 / np.pi, 360)

    num_Ks = 5
    K_contributions = np.zeros(num_Ks)
    missile_contributions = np.zeros((num_Ks, 3))
    release_po_all = [np.zeros((3, 3)) for _ in range(num_Ks)]
    explositon_po_all = [np.zeros((3, 3)) for _ in range(num_Ks)]
    explosion_time_all = [np.zeros(3) for _ in range(num_Ks)]

    for K_idx in range(num_Ks):
        K_param = K_params[K_idx]
        e_K = np.array([np.cos(K_param['direction_angle']),
                          np.sin(K_param['direction_angle']), 0])

        release_po = np.zeros((3, 3))
        explositon_po = np.zeros((3, 3))
        explosion_time = np.zeros(3)

        for j in range(3):
            release_po[j, :] = r_Ks0[K_idx, :] + K_param['speed'] * K_param['release_time'][j] * e_K
            explosion_time[j] = K_param['explosion_time'][j]
        
            explositon_po[j, :] = (release_po[j, :] +
                                   K_param['speed'] * K_param['delay_time'][j] * e_K -
                                   0.5 * g * (K_param['delay_time'][j] ** 2) * np.array([0, 0, 1]))

        release_po_all[K_idx] = release_po
        explositon_po_all[K_idx] = explositon_po
        explosion_time_all[K_idx] = explosion_time

    for K_idx in range(num_Ks):
        temp_K_params = [{} for _ in range(num_Ks)]
        for i in range(num_Ks):
            if i == K_idx:
                temp_K_params[i] = K_params[i].copy()
            else:
                temp_K_params[i] = {
                    'direction_angle': 0,
                    'direction_angle_deg': 0,
                    'speed': 100,
                    'release_time': np.array([1000, 1000, 1000]),
                    'delay_time': np.array([1, 1, 1]),
                    'explosion_time': np.array([1001, 1001, 1001])
                }
            
        total_contribution, missile_times, _ = evaluate_combined_solution(
            temp_K_params, r_missile0, e_missiles, v_missile,
            r_Ks0, g, R_cloud, v_sink, shadow_t,
            keypoints, t_range, dt
        )
        K_contributions[K_idx] = total_contribution
        missile_contributions[K_idx, :] = missile_times

    total_time, missile_times, effective_intervals = evaluate_combined_solution(
        K_params, r_missile0, e_missiles, v_missile,
        r_Ks0, g, R_cloud, v_sink, shadow_t,
        keypoints, t_range, dt
    )
    
    for i in range(3):
        print(f"导弹M{i + 1}有效遮蔽时间: {missile_times[i]:.2f} 秒")
        if effective_intervals[i].size > 0:
            print("  有效遮蔽区间：")
            for j in range(effective_intervals[i].shape[0]):
                start = effective_intervals[i][j, 0]
                end = effective_intervals[i][j, 1]
                print(f"    {start:.2f}秒 - {end:.2f}秒 (持续{end - start:.2f}秒)")
        else:
            print("  无有效遮蔽区间")

    for K_idx in range(num_Ks):
        K_param = K_params[K_idx]
        for j in range(3):
            flare_params = (
                explositon_po_all[K_idx][j, :],
                explosion_time_all[K_idx][j],
            )
            flare_t = evaluate_single_flare_contribution(
                flare_params, r_missile0, e_missiles, v_missile,
                g, R_cloud, v_sink, shadow_t,
                keypoints, t_range, dt
            )
            print(f"\n无人机FY{K_idx + 1} 烟幕弹 #{j + 1}:")
            print(f"  对导弹M1的遮蔽时间: {flare_t[0]:.2f} 秒")
            print(f"  对导弹M2的遮蔽时间: {flare_t[1]:.2f} 秒")
            print(f"  对导弹M3的遮蔽时间: {flare_t[2]:.2f} 秒")

    for i in range(num_Ks):
        print(f"\n无人机FY{i + 1}:")
        print(
            f"  飞行方向角: {K_params[i]['direction_angle_deg']:.2f} 度 ({K_params[i]['direction_angle']:.2f} 弧度)")
        print(f"  飞行速度: {K_params[i]['speed']:.2f} m/s")
        print(
            f"  对各导弹的贡献: M1={missile_contributions[i, 0]:.2f}s, M2={missile_contributions[i, 1]:.2f}s, M3={missile_contributions[i, 2]:.2f}s")

        for j in range(max_flares_per_K):
            print(f"  烟幕干扰弹 #{j + 1}:")
            print(f"    投放时间: {K_params[i]['release_time'][j]:.2f} s")
            print(f"    延迟时间: {K_params[i]['delay_time'][j]:.2f} s")
            print(f"    起爆时间: {K_params[i]['explosion_time'][j]:.2f} s")
            print(
                f"    投放点坐标: ({release_po_all[i][j, 0]:.2f}, {release_po_all[i][j, 1]:.2f}, {release_po_all[i][j, 2]:.2f}) m")
            print(
                f"    起爆点坐标: ({explositon_po_all[i][j, 0]:.2f}, {explositon_po_all[i][j, 1]:.2f}, {explositon_po_all[i][j, 2]:.2f}) m")

        print(f"  总贡献的有效遮蔽时间: {K_contributions[i]:.2f} s")

def main_independent_optimization():
    g = 9.8
    v_missile = 300
    R_cloud = 10
    v_sink = 3
    shadow_t = 20
    cylinder_radius = 7
    cylinder_height = 10
    min_interval = 1
    max_flares_per_K = 3

    r_M1_0 = np.array([20000, 0, 2000])
    r_M2_0 = np.array([19000, 600, 2100])
    r_M3_0 = np.array([18000, -600, 1900])
    r_missile0 = np.vstack([r_M1_0, r_M2_0, r_M3_0])

    O = np.array([0, 0, 0])
    T_base = np.array([0, 200, 0])

    r_FY1_0 = np.array([17800, 0, 1800])
    r_FY2_0 = np.array([12000, 1400, 1400])
    r_FY3_0 = np.array([6000, -3000, 700])
    r_FY4_0 = np.array([11000, 2000, 1800])
    r_FY5_0 = np.array([13000, -2000, 1300])
    r_Ks0 = np.vstack([r_FY1_0, r_FY2_0, r_FY3_0, r_FY4_0, r_FY5_0])

    e_M1 = (O - r_M1_0) / np.linalg.norm(O - r_M1_0)
    e_M2 = (O - r_M2_0) / np.linalg.norm(O - r_M2_0)
    e_M3 = (O - r_M3_0) / np.linalg.norm(O - r_M3_0)
    e_missiles = np.vstack([e_M1, e_M2, e_M3])
    t_start = 0
    t_end = 100
    dt = 0.1
    t_range = np.arange(t_start, t_end + dt, dt)
    n_t = len(t_range)
    
    n_points = 2
    keypoints = np.zeros((2 * n_points + 2, 3))
    keypoints[0, :] = T_base
    keypoints[1, :] = T_base + np.array([0, 0, cylinder_height])
    for i in range(1, n_points + 1):
        angle = 2 * np.pi * (i - 1) / n_points
        keypoints[2 + i - 1, :] = T_base + cylinder_radius * np.array([np.cos(angle), np.sin(angle), 0])
    for i in range(1, n_points + 1):
        angle = 2 * np.pi * (i - 1) / n_points
        keypoints[2 + n_points + i - 1, :] = T_base + cylinder_radius * np.array(
            [np.cos(angle), np.sin(angle), 0]) + np.array([0, 0, cylinder_height])

    dist_to_target = [np.linalg.norm(r_M1_0 - T_base),
                      np.linalg.norm(r_M2_0 - T_base),
                      np.linalg.norm(r_M3_0 - T_base)]
    est_flight_time = np.array(dist_to_target) / v_missile
    print(f"估计导弹飞行时间(M1, M2, M3): {est_flight_time} 秒")
    
    effective_time_window = [0, np.max(est_flight_time) * 1.2]
    max_release_t = effective_time_window[1] - 5 

    vars_per_K = 8
    num_Ks = 5

    K_opt_params = [{} for _ in range(num_Ks)]
    K_contributions = np.zeros(num_Ks)
    missile_contributions = np.zeros((num_Ks, 3))
    release_po_all = [np.zeros((3, 3)) for _ in range(num_Ks)]
    explositon_po_all = [np.zeros((3, 3)) for _ in range(num_Ks)]
    explosion_time_all = [np.zeros(3) for _ in range(num_Ks)]
    
    num_runs = 5
    
    for K_idx in range(num_Ks):
        print(f"======== 开始优化无人机 {K_idx + 1} ========")
        best_fval = -np.inf 
        best_x_opt = None

        for run in range(num_runs):
            print(f"-- 运行 {run + 1} / {num_runs} --")
            lb = np.array([
                0, 
                70, 
                0, 
                min_interval,
                min_interval, 
                0.0, 
                0.0, 
                0.0 
            ])
            ub = np.array([
                2 * np.pi, 
                140,
                max_release_t, 
                20,  
                20,
                10.0,  
                10.0,  
                10.0  
            ])

            def obj_fun(x):
                return -evaluate_single_K(
                    x, K_idx, r_missile0, e_missiles, v_missile,
                    r_Ks0, g, R_cloud, v_sink, shadow_t,
                    keypoints, t_range, dt
                )

            start_time = time.time()
            result = differential_evolution(
                obj_fun, bounds=list(zip(lb, ub)),
                popsize=20, maxiter=50, disp=False
            )
            optimization_time = time.time() - start_time
            print(f"优化完成，耗时: {optimization_time:.2f} 秒")

            current_solution = -result.fun
            if current_solution > best_fval:
                best_fval = current_solution
                best_x_opt = result.x
                print(f"发现更好的解，适应度值: {best_fval:.2f}")

        x_opt = best_x_opt
        K_param = single_solution_to_params(x_opt)
        K_opt_params[K_idx] = K_param

        e_K = np.array([np.cos(K_param['direction_angle']),
                          np.sin(K_param['direction_angle']), 0])
        release_po = np.zeros((3, 3))
        explositon_po = np.zeros((3, 3))
        explosion_time = np.zeros(3)

        for j in range(3):
            release_po[j, :] = r_Ks0[K_idx, :] + K_param['speed'] * K_param['release_time'][j] * e_K
            explosion_time[j] = K_param['explosion_time'][j]
            explositon_po[j, :] = (release_po[j, :] +
                                   K_param['speed'] * K_param['delay_time'][j] * e_K -
                                   0.5 * g * (K_param['delay_time'][j] ** 2) * np.array([0, 0, 1]))

        release_po_all[K_idx] = release_po
        explositon_po_all[K_idx] = explositon_po
        explosion_time_all[K_idx] = explosion_time

        total_contribution, missile_times, _ = evaluate_single_K_contribution(
            K_param, K_idx, r_missile0, e_missiles, v_missile,
            r_Ks0, g, R_cloud, v_sink, shadow_t,
            keypoints, t_range, dt
        )
        K_contributions[K_idx] = total_contribution
        missile_contributions[K_idx, :] = missile_times

        print(f"无人机{K_idx + 1}总贡献: {total_contribution:.2f} 秒")
        print(f"对导弹M1贡献: {missile_times[0]:.2f} 秒")
        print(f"对导弹M2贡献: {missile_times[1]:.2f} 秒")
        print(f"对导弹M3贡献: {missile_times[2]:.2f} 秒")
        print(f"飞行方向角: {K_param['direction_angle_deg']:.2f} 度")
        print(f"飞行速度: {K_param['speed']:.2f} m/s")
        for j in range(3):
            print(f"烟幕干扰弹 #{j + 1}:")
            print(f"  投放时间: {K_param['release_time'][j]:.2f} s")
            print(f"  延迟时间: {K_param['delay_time'][j]:.2f} s")
            print(f"  起爆时间: {K_param['explosion_time'][j]:.2f} s")
            print(f"  投放点坐标: ({release_po[j, 0]:.2f}, {release_po[j, 1]:.2f}, {release_po[j, 2]:.2f}) m")
            print(f"  起爆点坐标: ({explositon_po[j, 0]:.2f}, {explositon_po[j, 1]:.2f}, {explositon_po[j, 2]:.2f}) m")
            
        for j in range(3):
            flare_params = (
                explositon_po[j, :],
                explosion_time[j],
            )
            flare_t = evaluate_single_flare_contribution(
                flare_params, r_missile0, e_missiles, v_missile,
                g, R_cloud, v_sink, shadow_t,
                keypoints, t_range, dt
            )
            print(f"  烟幕弹 #{j + 1} 对导弹M1的遮蔽时间: {flare_t[0]:.2f} 秒")
            print(f"  烟幕弹 #{j + 1} 对导弹M2的遮蔽时间: {flare_t[1]:.2f} 秒")
            print(f"  烟幕弹 #{j + 1} 对导弹M3的遮蔽时间: {flare_t[2]:.2f} 秒")

    total_time, missile_times, effective_intervals = evaluate_combined_solution(
        K_opt_params, r_missile0, e_missiles, v_missile,
        r_Ks0, g, R_cloud, v_sink, shadow_t,
        keypoints, t_range, dt
    )
    print(f"总有效遮蔽时间: {total_time:.2f} 秒")
    for i in range(3):
        print(f"导弹M{i + 1}有效遮蔽时间: {missile_times[i]:.2f} 秒")
        if effective_intervals[i].size > 0:
            print("  有效遮蔽区间：")
            for j in range(effective_intervals[i].shape[0]):
                start = effective_intervals[i][j, 0]
                end = effective_intervals[i][j, 1]
                print(f"    {start:.2f}秒 - {end:.2f}秒 (持续{end - start:.2f}秒)")
        else:
            print("  无有效遮蔽区间")

    for i in range(num_Ks):
        print(f"\n无人机FY{i + 1}:")
        print(
            f"  飞行方向角: {K_opt_params[i]['direction_angle_deg']:.2f} 度 ({K_opt_params[i]['direction_angle']:.2f} 弧度)")
        print(f"  飞行速度: {K_opt_params[i]['speed']:.2f} m/s")
        print(
            f"  对各导弹的贡献: M1={missile_contributions[i, 0]:.2f}s, M2={missile_contributions[i, 1]:.2f}s, M3={missile_contributions[i, 2]:.2f}s")

        for j in range(max_flares_per_K):
            print(f"  烟幕干扰弹 #{j + 1}:")
            print(f"    投放时间: {K_opt_params[i]['release_time'][j]:.2f} s")
            print(f"    延迟时间: {K_opt_params[i]['delay_time'][j]:.2f} s")
            print(f"    起爆时间: {K_opt_params[i]['explosion_time'][j]:.2f} s")
            print(
                f"    投放点坐标: ({release_po_all[i][j, 0]:.2f}, {release_po_all[i][j, 1]:.2f}, {release_po_all[i][j, 2]:.2f}) m")
            print(
                f"    起爆点坐标: ({explositon_po_all[i][j, 0]:.2f}, {explositon_po_all[i][j, 1]:.2f}, {explositon_po_all[i][j, 2]:.2f}) m")

        print(f"  总贡献的有效遮蔽时间: {K_contributions[i]:.2f} s")

def isLineSegmentIntersectSphere(p1, p2, center, radius):
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


def evaluate_combined_solution_with_status(K_params, r_missile0, e_missiles, v_missile,
                                           r_Ks0, g, R_cloud, v_sink, shadow_t,
                                           keypoints, t_range, dt):
    num_Ks = len(K_params)
    n_t = len(t_range)
    num_missiles = 3

    e_Ks = np.zeros((num_Ks, 3))
    for i in range(num_Ks):
        e_Ks[i, :] = np.array([np.cos(K_params[i]['direction_angle']),
                                 np.sin(K_params[i]['direction_angle']), 0])

    release_po = [np.zeros((3, 3)) for _ in range(num_Ks)]
    explositon_po = [np.zeros((3, 3)) for _ in range(num_Ks)]
    for i in range(num_Ks):
        for j in range(3):
            release_po[i][j, :] = (r_Ks0[i, :] +
                                    K_params[i]['speed'] * K_params[i]['release_time'][j] * e_Ks[i, :])
            explositon_po[i][j, :] = (release_po[i][j, :] +
                                      K_params[i]['speed'] * K_params[i]['delay_time'][j] * e_Ks[i, :] -
                                      0.5 * g * (K_params[i]['delay_time'][j] ** 2) * np.array([0, 0, 1]))

    r_missiles = np.zeros((n_t, num_missiles, 3))
    r_clouds = [[np.zeros((n_t, 3)) for _ in range(3)] for __ in range(num_Ks)]
    is_effective = np.zeros((n_t, num_missiles))
    has_cloud = np.zeros((n_t, num_Ks, 3))

    for i in range(n_t):
        t = t_range[i]
        for j in range(num_missiles):
            r_missiles[i, j, :] = r_missile0[j, :] + v_missile * t * e_missiles[j, :]

        for j in range(num_Ks):
            for k in range(3):
                if (t >= K_params[j]['explosion_time'][k]) and (
                        t <= K_params[j]['explosion_time'][k] + shadow_t):
                    has_cloud[i, j, k] = 1
                    cloud_center = (explositon_po[j][k, :] -
                                    v_sink * (t - K_params[j]['explosion_time'][k]) * np.array([0, 0, 1]))
                    r_clouds[j][k][i, :] = cloud_center

        for j in range(num_missiles):
            all_points_covered = True
            for k in range(keypoints.shape[0]): 
                point_covered = False
                for m in range(num_Ks):
                    for n in range(3):
                        if has_cloud[i, m, n] == 1:
                            cloud_center = r_clouds[m][n][i, :]
                            if isLineSegmentIntersectSphere(
                                    r_missiles[i, j, :].reshape(-1, 1),
                                    keypoints[k, :].reshape(-1, 1),
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
            effective_intervals[j] = np.array([])
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

def evaluate_combined_solution(K_params, r_missile0, e_missiles, v_missile,
                               r_Ks0, g, R_cloud, v_sink, shadow_t,
                               keypoints, t_range, dt):
    total_time, missile_times, effective_intervals, _ = evaluate_combined_solution_with_status(
        K_params, r_missile0, e_missiles, v_missile,
        r_Ks0, g, R_cloud, v_sink, shadow_t,
        keypoints, t_range, dt
    )
    return total_time, missile_times, effective_intervals


def single_solution_to_params(x):
    direction_angle = x[0]
    direction_angle_deg = np.mod(direction_angle * 180 / np.pi, 360)
    speed = x[1]

    release_time = np.zeros(3)
    release_time[0] = x[2]
    release_time[1] = release_time[0] + x[3]
    release_time[2] = release_time[1] + x[4]

    delay_time = x[5:8]

    explosion_time = release_time + delay_time

    return {
        'direction_angle': direction_angle,
        'direction_angle_deg': direction_angle_deg,
        'speed': speed,
        'release_time': release_time,
        'delay_time': delay_time,
        'explosion_time': explosion_time
    }

def evaluate_single_K(x, K_idx, r_missile0, e_missiles, v_missile,
                        r_Ks0, g, R_cloud, v_sink, shadow_t,
                        keypoints, t_range, dt):
    K_param = single_solution_to_params(x)
    total_contribution, _, _ = evaluate_single_K_contribution(
        K_param, K_idx, r_missile0, e_missiles, v_missile,
        r_Ks0, g, R_cloud, v_sink, shadow_t,
        keypoints, t_range, dt
    )
    return total_contribution

def evaluate_single_K_contribution(K_param, K_idx, r_missile0, e_missiles, v_missile,
                                     r_Ks0, g, R_cloud, v_sink, shadow_t,
                                     keypoints, t_range, dt):
    num_Ks = 5
    temp_K_params = [{} for _ in range(num_Ks)]
    for i in range(num_Ks):
        if i == K_idx:
            temp_K_params[i] = K_param.copy()
        else:
            temp_K_params[i] = {
                'direction_angle': 0,
                'direction_angle_deg': 0,
                'speed': 100,
                'release_time': np.array([1000, 1000, 1000]),
                'delay_time': np.array([1, 1, 1]),
                'explosion_time': np.array([1001, 1001, 1001])
            }
            
    total_contribution, missile_times, _ = evaluate_combined_solution(
        temp_K_params, r_missile0, e_missiles, v_missile,
        r_Ks0, g, R_cloud, v_sink, shadow_t,
        keypoints, t_range, dt
    )
    return total_contribution, missile_times, _


def evaluate_single_flare_contribution(flare_params, r_missile0, e_missiles, v_missile,
                                       g, R_cloud, v_sink, shadow_t,
                                       keypoints, t_range, dt):
    num_missiles = 3
    n_t = len(t_range)
    flare_explositon_po, flare_explosion_time = flare_params

    is_effective = np.zeros((n_t, num_missiles))

    for i in range(n_t):
        t = t_range[i]
        missile_positions = np.zeros((num_missiles, 3))
        for j in range(num_missiles):
            missile_positions[j, :] = r_missile0[j, :] + v_missile * t * e_missiles[j, :]

        if (t >= flare_explosion_time) and (t <= flare_explosion_time + shadow_t):
            cloud_center = flare_explositon_po - v_sink * (t - flare_explosion_time) * np.array([0, 0, 1])
            for j in range(num_missiles):
                all_points_covered = True
                for k in range(keypoints.shape[0]):
                    if not isLineSegmentIntersectSphere(
                            missile_positions[j, :].reshape(-1, 1),
                            keypoints[k, :].reshape(-1, 1),
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

    print("=" * 50)
    print("开始执行综合最优参数评估")
    print("=" * 50)
    main_combined_evaluation()

    print("\n" + "=" * 50)
    print("开始执行无人机独立优化")
    print("=" * 50)
    main_independent_optimization()

    tee.close()
    print(f"\n结果已保存到 {filename}")