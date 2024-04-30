import numpy as np
import argparse
from geometry import *
from threejs_group import *
import os
from car_prop import CarRobot

class ParticleFilter:
    def __init__(self, start, landmark_pos, landmarks_data, noise_level, num_particles, viz_out):
        self.start = start
        self.landmarks = landmark_pos
        self.landmark_data = landmarks_data
        self.car_robot = CarRobot(q0=self.start, actuation_noise=None, odometry_noise=None, observation_noise=None,
                                  viz_out=viz_out, landmarks=self.landmarks)
       
        self.odometry_noise = self.car_robot.odometry_noise_model[noise_level]
        self.observation_noise = self.car_robot.observation_noise_model[noise_level]
        self.actuation_noise = self.car_robot.actuation_noise_model[noise_level]
       
        self.num_particles = num_particles
        self.viz_out = viz_out
       
        # Initialize particles
        self.particles = np.zeros((self.num_particles, 3))
        self.particles[:, :2] = self.start[:2]
        self.particles[:, 2] = np.random.uniform(-np.pi, np.pi, self.num_particles)
        self.weights = np.ones(self.num_particles) / self.num_particles
        self.particle_observations = [[] for _ in range(self.num_particles)]

        self.visualize_landmarks()

    def resample(self):
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def update_particles(self, odometry):
        # Update particles based on odometry
        for i, particle in enumerate(self.particles):
            x, y, theta = particle
            v, phi = odometry
            if self.odometry_noise is not None:
                v = np.random.normal(v, np.sqrt(self.odometry_noise[0]))
                phi = np.random.normal(phi, np.sqrt(self.odometry_noise[1]))
            x_dot = v * np.cos(theta)
            y_dot = v * np.sin(theta)
            theta_dot = (v / self.car_robot.L) * np.tan(phi)
            self.particles[i] = [x + x_dot * self.car_robot.dt, y + y_dot * self.car_robot.dt, (theta + theta_dot * self.car_robot.dt) % (2 * np.pi) - np.pi]

    def get_all_landmark_observations(self, particles):
        landmark_observations = []
        for particle in particles:
            x, y, theta = particle
            particle_observations = []
            for landmark in self.landmarks:
                d = np.sqrt((landmark[0] - x)**2 + (landmark[1] - y)**2)
                alpha = np.arctan2(landmark[1]-y,landmark[0]-x) - theta
                if self.observation_noise:
                    d = np.random.normal(d, np.sqrt(self.observation_noise[0]))
                    alpha = np.random.normal(alpha, np.sqrt(self.observation_noise[1]))
                particle_observations.extend([d, alpha])
            landmark_observations.append(particle_observations)
        return landmark_observations

    def update_weights(self, landmark_observations):
        # Update weights based on landmark observations
        new_weights = []
        for j, particle in enumerate(self.particles):
            x, y, theta = particle
            weight = 1
            for i in range(len(self.landmarks)):
                d, alpha = landmark_observations[j][i*2], landmark_observations[j][i*2+1]
                expected_d = np.sqrt((self.landmarks[i][0] - x)**2 + (self.landmarks[i][1] - y)**2)
                expected_alpha = np.arctan2(self.landmarks[i][1] - y, self.landmarks[i][0] - x) - theta
                expected_alpha = (expected_alpha + np.pi) % (2 * np.pi) - np.pi
                weight *= np.exp(-((d - expected_d)**2 / (2 * self.observation_noise[0]**2) + (alpha - expected_alpha)**2 / (2 * self.observation_noise[1]**2)))
            new_weights.append(weight)
        total_weight = sum(new_weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in new_weights]
        else:
            # Handle the case where all weights are zero
            self.weights = [1.0 / self.num_particles] * self.num_particles

    def get_estimated_pose(self):
        # Compute the weighted mean of the particles as the estimated pose
        x = np.average(self.particles[:, 0], weights=self.weights)
        y = np.average(self.particles[:, 1], weights=self.weights)
        theta = np.arctan2(np.sum(np.sin(self.particles[:, 2]) * self.weights), np.sum(np.cos(self.particles[:, 2]) * self.weights))
        return [x, y, theta]

    def add_particle_info(self, t):
        # Visualize the particles
        for i, p in enumerate(self.particles):
            self.particle_observations[i].append([t, [p[0], p[1], 0.5], [1, 0, 0, 0], "#088F8F"])

    def visualize_particles(self):
        for i in range(self.num_particles):
            geom = sphere('part_' + str(i), 0.5, self.particle_observations[i][0][1], self.particle_observations[i][0][2])
            self.viz_out.add_animation(geom, self.particle_observations[i])
        return self.viz_out

    def visualize_landmarks(self):
        blue = "0x0000ff"
        for i, [x, y] in enumerate(self.landmarks):
            geom = sphere('obs' + str(i), 0.5, [x, y, 0.5], [1, 0, 0, 0])
            self.viz_out.add_obstacle(geom, blue)
        return

# ... (rest of the file remains unchanged) # ... (previous ParticleFilter class code)

if __name__ == "__main__":
    viz_out = threejs_group(js_dir="../js")
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, required=True)
    parser.add_argument("--odometry", type=str, required=True)
    parser.add_argument("--landmarks", type=str, required=True)
    parser.add_argument("--plan", type=str, required=True)
    parser.add_argument("--num_particles", type=int, required=True)
    args = parser.parse_args()

    red = "0xff0000"
    gt_col = "#3CB371"
    estimate_col = "#964B00"
    # Load data
    landmark_pos = np.loadtxt(args.map)
    start = np.loadtxt(args.plan, max_rows=1)
    landmarks_data = np.loadtxt(args.landmarks, skiprows=1)
    odometry = np.loadtxt(args.odometry)
    problem_id = args.odometry.split('_')[-2]
    noise_level = args.odometry.split('_')[-1][0]
    pf = ParticleFilter(start, landmark_pos, landmarks_data, noise_level, args.num_particles, viz_out)
   
    pose_estimates = []
    # Run particle filter
    for t in range(len(odometry)):
        pf.update_particles(odometry[t])
        landmark_observations = pf.get_all_landmark_observations(pf.particles)
        pf.update_weights(landmark_observations)
        pf.resample()
        estimated_pose = pf.get_estimated_pose()
        pose_estimates.append(estimated_pose)
        pf.add_particle_info(t)

    np.savetxt(f"./py160-hp580/data/pf_estimates_{problem_id}_{noise_level}_{args.num_particles}.txt", pose_estimates)
    gt_trajectory = np.loadtxt(f'./py160-hp580/data/gt_{problem_id}_{noise_level}.txt')
    estimates_trajectory = [[x, y, 0.5] for x, y, _ in pose_estimates]
    gt_traj = [[x, y, 0.5] for x, y, _ in gt_trajectory]

    pf.car_robot.visualize_linepath_color(np.array(estimates_trajectory), estimate_col)
    pf.car_robot.visualize_given_trajectory_name(estimates_trajectory, "est_traj")
    pf.car_robot.visualize_linepath_color(np.array(gt_traj), gt_col)
    pf.car_robot.visualize_given_trajectory_name(gt_traj, "gt_traj")

    pf.visualize_particles()
    viz_out.to_html("particle_filter.html", "out/")