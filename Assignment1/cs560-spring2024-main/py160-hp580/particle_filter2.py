import numpy as np
import argparse
from geometry import *
from threejs_group import *
import os
from car_prop import CarRobot
from dead_reckoning import DeadReckoning

class ParticleFilter:
    def __init__(self, start, landmark_pos , landmarks_data ,  noise_level, num_particles, viz_out):
        self.start = start
        self.landmarks = landmark_pos
        self.landmark_data = landmarks_data
        self.car_robot = CarRobot(q0=self.start, actuation_noise=None, odometry_noise=None, observation_noise=None, 
                                  viz_out=viz_out, landmarks= None)
        
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
        cumulative_weights = np.cumsum(self.weights)
        indices = np.searchsorted(cumulative_weights, np.random.rand(self.num_particles))
        self.particles = [self.particles[i] for i in indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def update_particles(self, odometry):
        # Update particles based on odometry
        new_particles = []
        for particle in self.particles:
            x, y, theta = particle
            v, phi = odometry
            if self.odometry_noise is not None:
                if v != 0 :
                    v = np.random.normal(v, np.sqrt(self.odometry_noise[0]))
                if phi != 0:
                    phi = np.random.normal(phi, np.sqrt(self.odometry_noise[1]))
            x_dot = v * np.cos(theta)
            y_dot = v * np.sin(theta)
            theta_dot = (v / 1.5) * np.tan(phi)
            new_x = x + x_dot
            new_y = y + y_dot
            new_theta = (theta + theta_dot) % (2 * np.pi) - np.pi
            new_particles.append([new_x, new_y, new_theta])
        self.particles = new_particles

    def get_all_landmark_observations(self, particles):
        landmark_observations = []
        for particle in particles:
            x, y, theta = particle
            particle_observations = []
            for landmark in self.landmarks:
                d = np.sqrt((landmark[0] - x)**2 + (landmark[1] - y)**2) # ground truth
                alpha = np.arctan2(landmark[1]-y,landmark[0]-x) - theta  # ground truth 
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
            for i in range(int(len(landmark_observations[0])/2)):
                d,alpha = landmark_observations[j][i*2],landmark_observations[j][i*2+1]
                if self.observation_noise is not None:
                    d = np.random.normal(d, np.sqrt(self.observation_noise[0]))
                    alpha = np.random.normal(alpha, np.sqrt(self.observation_noise[1]))
                expected_d = np.sqrt((self.landmarks[0] - x)**2 + (self.landmarks[1] - y)**2)
                expected_alpha = np.arctan2(self.landmarks[1] - y, self.landmarks[0] - x) - theta
                expected_alpha = (expected_alpha + np.pi) % (2 * np.pi) - np.pi
                weight *= np.exp(-((d - expected_d)**2 / (2 * self.observation_noise[0]**2) +
                                  (alpha - expected_alpha)**2 / (2 * self.observation_noise[1]**2)))
            new_weights.append(weight)
        self.weights = np.array(new_weights) / np.sum(new_weights)

    def get_estimated_pose(self):
        # Compute the weighted mean of the particles as the estimated pose
        x = np.sum([p[0] * w for p, w in zip(self.particles, self.weights)])
        y = np.sum([p[1] * w for p, w in zip(self.particles, self.weights)])
        theta = np.sum([p[2] * w for p, w in zip(self.particles, self.weights)])
        return [x, y, theta]

    def add_particle_info(self, t):
        # Visualize the particles
        for i,p in enumerate(self.particles):
            self.particle_observations[i].append([t , [p[0] ,p[1], 0.5] , [1,0,0,0] , "#088F8F"])
    
    def visualize_particles(self):
        for i in range(self.num_particles):
            geom = sphere('part_'+str(i) , 0.5 ,self.particle_observations[i][0][1] , self.particle_observations[i][0][2])
            self.viz_out.add_animation(geom , self.particle_observations[i] )
        return self.viz_out

    def visualize_landmarks(self):
        blue="0x0000ff"
        for i,[x,y] in enumerate(self.landmarks):
            geom = sphere('obs'+str(i) , 0.5 , [x,y,0.5] , [1,0,0,0] )
            self.viz_out.add_obstacle(geom , blue)
        return 
    


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
    # Load data
    landmark_pos = np.loadtxt(args.map)
    start = np.loadtxt(args.plan, max_rows=1)
    landmarks_data = np.loadtxt(args.landmarks, skiprows=1)
    odometry = np.loadtxt(args.odometry)
    problem_id = args.odometry.split('_')[-2]
    noise_level = args.odometry.split('_')[-1][0]
    pf = ParticleFilter(start, landmark_pos , landmarks_data ,  noise_level, args.num_particles, viz_out)
    dead_reckoning =  DeadReckoning(None , args.map, args.odometry, args.landmarks, args.plan)
    dead_reckoning_states = dead_reckoning.dead_reckoning_trajectory(dead_reckoning.odometry_data , dead_reckoning.start_state)
    pf.car_robot.visualize_linepath_color( dead_reckoning_states, red ) 
    pf.car_robot.visualize_given_trajectory_name(dead_reckoning_states , "dead_reckon")
    
    # Run particle filter
    for t in range(len(odometry)):
        pf.update_particles(odometry[t])
        landmark_observations = pf.get_all_landmark_observations(pf.particles)
        pf.update_weights(landmark_observations)
        pf.resample()
        estimated_pose = pf.get_estimated_pose()
        pf.add_particle_info(t)

    pf.visualize_particles()
    viz_out.to_html("particle_filter.html", "out/")