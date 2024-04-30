import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import time
from geometry import *
from threejs_group import *
from car_prop import CarRobot
import copy 

class KalmanFilter:
    def __init__(self, start, landmarks, landmarks_data, odometry_data , actuation_noise, observation_noise, viz_out):
        self.start = start
        self.landmarks = landmarks
        self.landmark_data = landmarks_data
        self.odometry_data = odometry_data
        self.viz_out = viz_out
        self.L = 1.5
        self.actuation_noise = actuation_noise
        self.observation_noise = observation_noise

        self.state = self.start
        self.covariance = np.eye(3) * 0.1  # Initialize covariance based on expected uncertainty

        self.visualize_landmarks()

    @staticmethod
    def rotate_z_quaternion(theta):
        return [np.cos(theta/2), 0, 0, np.sin(theta/2)]

    def propagate_state(self, state, control , dt=0.1):
        x, y, theta = state
        v, phi = control
        x_new = x + v * np.cos(theta) * dt
        y_new = y + v * np.sin(theta) * dt
        theta_new = theta + (v / self.L) * np.tan(phi) * dt
        return np.array([x_new, y_new, theta_new])

    def jacobian_F(self, state, control, dt = 0.1):
        _, _, theta = state
        v = control[0]
        return np.array([
            [1, 0, -v * np.sin(theta) * dt],
            [0, 1, v * np.cos(theta) * dt],
            [0, 0, 1]
        ])

    def visualize_given_trajectory_name(self, trajectory , name):
        yellow = "0xFFFF00"
        car_trajectory = []
        for t in range(len(trajectory)):
            x,y,theta = trajectory[t]
            quat = self.rotate_z_quaternion(theta=theta) 
            car_trajectory.append([t, [x,y,0.5] , quat , yellow ])
        
        cube = box(name, 2,1,1, car_trajectory[0][1], car_trajectory[0][2])
        self.viz_out.add_animation(cube , car_trajectory)
        return self.viz_out 

    def observation_model(self, state, landmark):
        x, y, _ = state
        lx, ly = landmark
        dx, dy = lx - x, ly - y
        r = np.hypot(dx, dy)
        beta = np.arctan2(dy, dx)
        return np.array([r, beta])
    
    def jacobian_H(self, state, landmark):
        x, y, _ = state
        lx, ly = landmark
        dx, dy = lx - x, ly - y
        r2 = dx*2 + dy*2
        return np.array([[-dx / np.sqrt(r2), -dy / np.sqrt(r2), 0],[dy / r2, -dx / r2, -1]])

    def visualize_linepath_color(self, trajectory_path ,  color):  
        trajectory_path = copy.deepcopy(trajectory_path)
        trajectory_path[:,2] = 0.5
        self.viz_out.add_line(trajectory_path, color) 
        return self.viz_out 
    
    def ekf_predict(self, mu, sigma, u, dt , R):
        F = self.jacobian_F(mu, u, dt)
        mu_bar = self.propagate_state(mu, u, dt)
        sigma_bar = F @ sigma @ F.T + R
        return mu_bar, sigma_bar

    def ekf_update(self , mu_bar, sigma_bar, z, landmark, Q):
        H = self.jacobian_H(mu_bar, landmark)
        z_hat = self.observation_model(mu_bar, landmark)
        S = H @ sigma_bar @ H.T + Q
        K = sigma_bar @ H.T @ np.linalg.inv(S)
        z_diff = z - z_hat
        z_diff[1] = (z_diff[1] + np.pi) % (2 * np.pi) - np.pi  # Normalize bearing
        mu = mu_bar + K @ z_diff
        sigma = (np.eye(len(mu)) - K @ H) @ sigma_bar
        return mu, sigma

    def plot_estimation_vs_ground_truth(self, estimations, ground_truth):
        fig, ax = plt.subplots()
        est_xs, est_ys = zip(*[(est[0], est[1]) for est in estimations])
        true_xs, true_ys = zip(*[(true[0], true[1]) for true in ground_truth])
        ax.plot(est_xs, est_ys, 'r-', label='EKF Estimate')
        ax.plot(true_xs, true_ys, 'b-', label='Ground Truth')
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.legend()
        plt.title('EKF Localization vs. Ground Truth')
        plt.show()

    def run_kf(self):
        mu = np.array([0, 0, 0])  # Assume initial pose
        sigma = np.eye(3) * 0.1  # Initial uncertainty
        dt = 0.1
        R = np.diag([self.actuation_noise[0]*2, self.actuation_noise[0]*2, (self.actuation_noise[1] / 1.5)*2])
        Q = np.diag(self.observation_noise)        
        estimates = []
        for control_index, control in enumerate(self.odometry_data):
            mu, sigma = self.ekf_predict(mu, sigma, control, dt, R)
            estimates.append(mu.copy())
            current_measurements = self.landmark_data[control_index].reshape(-1, 2)
            for measurement, landmark in zip(current_measurements, landmarks):
                mu, sigma = self.ekf_update(mu, sigma, measurement, landmark, Q)
        return estimates

    def visualize_landmarks(self):
        blue = "0x0000ff"
        for i, [x, y] in enumerate(self.landmarks):
            geom = sphere('obs' + str(i), 0.5, [x, y, 0.5], [1, 0, 0, 0])
            self.viz_out.add_obstacle(geom, blue)
        return

if __name__ == "__main__":
    start_time = time.time()
    viz_out = threejs_group(js_dir="../js")
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, required=True)
    parser.add_argument("--odometry", type=str, required=True)
    parser.add_argument("--landmarks", type=str, required=True)
    parser.add_argument("--plan", type=str, required=True)
    args = parser.parse_args()

    red = "0xff0000"
    gt_col = "#3CB371"
    estimate_col = "#964B00"

    # Load data
    landmarks = np.loadtxt(args.map)
    start = np.loadtxt(args.plan, max_rows=1)
    landmarks_data = np.loadtxt(args.landmarks)
    odometry_data = np.loadtxt(args.odometry)

    problem_id = args.odometry.split('_')[-2]
    noise_level = args.odometry.split('_')[-1][0]
    gt_trajectory = np.loadtxt(f'./py160-hp580/data/gt_{problem_id}_{noise_level}.txt')


    car_robot = CarRobot(q0=start, actuation_noise=None, odometry_noise=None, observation_noise=None,
                                  viz_out=viz_out, landmarks=None)

    # odometry_noise = car_robot.odometry_noise_model[noise_level]
    observation_noise = car_robot.observation_noise_model[noise_level]
    actuation_noise = car_robot.actuation_noise_model[noise_level]
        
    kf = KalmanFilter(start, landmarks, landmarks_data, odometry_data, actuation_noise, observation_noise, viz_out)
    pose_estimates = kf.run_kf()

    np.savetxt(f"./py160-hp580/data/kf_estimates_{problem_id}_{noise_level}.txt", pose_estimates)
    gt_trajectory = np.loadtxt(f'./py160-hp580/data/gt_{problem_id}_{noise_level}.txt')

    # kf.viz_out.add_line(np.array(pose_estimates), estimate_col)
    # kf.viz_out.add_line(np.array(gt_trajectory), gt_col)

    # kf.visualize_linepath_color(np.array(pose_estimates), estimate_col)
    kf.visualize_given_trajectory_name(pose_estimates, "est_traj")
    # kf.visualize_linepath_color(np.array(gt_trajectory), gt_col)
    kf.visualize_given_trajectory_name(gt_trajectory, "gt_traj")
    
    kf.viz_out.to_html("kalman_filter.html", "out/")