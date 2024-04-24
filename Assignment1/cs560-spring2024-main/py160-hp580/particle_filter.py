import argparse  
import numpy as np  
from dead_reckoning import visualize_dead_reckoning
from data_collection import DataCollection  
from threejs_group import threejs_group  

def initialize_particles(num_particles, start, use_known_start=True):  
    particles = []  
    for _ in range(num_particles):  
        if use_known_start:  
            particle = start  
        else:  
            x = np.random.uniform(-50, 50)  
            y = np.random.uniform(-50, 50)  
            theta = np.random.uniform(-np.pi, np.pi)  
            particle = np.array([x, y, theta])  
        particles.append(particle)  
    return particles  

def update_particles(particles, odometry):  
    updated_particles = []  
    for particle in particles:  
        x, y, theta = particle  
        v, phi = odometry  
        theta_new = theta + v * phi * 0.1  
        x_new = x + v * np.cos(theta) * 0.1  
        y_new = y + v * np.sin(theta) * 0.1  
        updated_particles.append(np.array([x_new, y_new, theta_new]))  
    return updated_particles  

def calculate_weights(particles, landmark_observations, landmarks, observation_noise):  
    weights = []  
    for particle in particles:  
        weight = 1.0  
        for obs, landmark in zip(landmark_observations, landmarks):  
            d, alpha = obs  
            x, y, theta = particle  
            x_l, y_l = landmark  
            d_hat = np.sqrt((x_l - x)**2 + (y_l - y)**2)  
            alpha_hat = np.arctan2(y_l - y, x_l - x) - theta  
            weight *= np.exp(-0.5 * ((d - d_hat)**2 / observation_noise[0]**2 + (alpha - alpha_hat)**2 / observation_noise[1]**2))  
        weights.append(weight)  
    return np.array(weights)  

def resample_particles(particles, weights):  
    num_particles = len(particles)  
    indices = np.arange(num_particles)  
    resampled_indices = np.random.choice(indices, size=num_particles, replace=True, p=weights/np.sum(weights))  
    resampled_particles = [particles[i] for i in resampled_indices]  
    return resampled_particles  


def load_data(filename):  
    data = []  
    with open(filename, 'r') as file:  
        for line in file:  
            line = line.strip().replace('(', '').replace(')', '').replace(',', '').split()  
            data.append([float(x) for x in line])  
    return np.array(data)  


# Particle Filter Implementation  
def particle_filter(odometry_data, landmark_data, num_particles, start, landmarks, observation_noise):  
    particles = initialize_particles(num_particles, start)  
    estimated_poses = [start]  
      
    for odometry, landmark_observations in zip(odometry_data, landmark_data):  
        particles = update_particles(particles, odometry)  
        weights = calculate_weights(particles, landmark_observations, landmarks, observation_noise)  
        particles = resample_particles(particles, weights)  
        estimated_poses.append(np.mean(particles, axis=0))  
      
    return estimated_poses  
  
if __name__ == "__main__":  
    parser = argparse.ArgumentParser()  
    parser.add_argument("--map", type=str, required=True)  
    parser.add_argument("--odometry", type=str, required=True)  
    parser.add_argument("--landmarks", type=str, required=True)  
    parser.add_argument("--plan", type=str, required=True)  
    parser.add_argument("--num_particles", type=int, required=True)  
    args = parser.parse_args()  
    viz_out = threejs_group(js_dir="../js")
    # Load the data 
    ground_truth_data = load_data('/data/gt_1.0_L.txt')  
    odometry_data = load_data(args.odometry)  
    landmark_data = load_data(args.landmarks)  
    start = np.array([5, 25, 0.5])  
    num_particles = args.num_particles  
    observation_noise = [0.1, 0.1]  
    landmarks = np.loadtxt("landmark_0.txt", skiprows=1)  
  
    # Call the Particle Filter function  
    estimated_poses = particle_filter(odometry_data, landmark_data, num_particles, start, landmarks, observation_noise)  
  
    # Visualize the results  
    visualize_dead_reckoning(viz_out, ground_truth_data, estimated_poses)  
