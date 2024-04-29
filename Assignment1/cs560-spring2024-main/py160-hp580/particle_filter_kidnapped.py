import argparse  
import numpy as np  
from geometry import *  
from threejs_group import *  
from car_prop import CarRobot  
from particle_filter import ParticleFilter

class UnknownStartParticleFilter(ParticleFilter):  
    def __init__(self, *args, **kwargs):  
        super().__init__(*args, **kwargs)  
  
    def initialize_particles(self):  
        particles = np.zeros((self.num_particles, 3))  
  
        # Randomize particles' positions within the environment bounds  
        x_range = (-50, 50)  
        y_range = (-50, 50)  
        particles[:, 0] = np.random.uniform(x_range[0], x_range[1], self.num_particles)  
        particles[:, 1] = np.random.uniform(y_range[0], y_range[1], self.num_particles)  
  
        # Randomize particles' orientations  
        particles[:, 2] = np.random.uniform(-np.pi, np.pi, self.num_particles)  
  
        return particles  
    
    def load_start_state(self, filename):  
        with open(filename, 'r') as file:  
            for line in file:  
                line = line.strip().split()  
                line = [float(x) for x in line]  
                break  
        return line
    
    def motion_update(self, odometry):  
        v, phi = odometry  
        dt = 0.1  
        noise_scale = 0.1  # Adjust based on your model's accuracy  
  
        theta = self.particles[:, 2]  
        theta_dot = (v / 1.5) * np.tan(phi)  # Assuming L = 1.5  
  
        # Adding noise to the motion model  
        delta_x = (v + np.random.normal(0, noise_scale, self.num_particles)) * np.cos(theta) * dt  
        delta_y = (v + np.random.normal(0, noise_scale, self.num_particles)) * np.sin(theta) * dt  
        delta_theta = theta_dot * dt + np.random.normal(0, noise_scale, self.num_particles)  
  
        # Update particles' positions and orientations  
        self.particles[:, 0] += delta_x  
        self.particles[:, 1] += delta_y  
        self.particles[:, 2] += delta_theta  
        # Normalize angles  
        self.particles[:, 2] = (self.particles[:, 2] + np.pi) % (2 * np.pi) - np.pi  
  
    def measurement_update(self, landmark_obs):  
        for i, landmark in enumerate(self.map_data):  
            # Extract landmark observation for the current landmark  
            d, alpha = landmark_obs[i * 2:i * 2 + 2]  
  
            # Calculate expected distance and angle for each particle to the current landmark  
            dx = landmark[0] - self.particles[:, 0]  
            dy = landmark[1] - self.particles[:, 1]  
            exp_d = np.sqrt(dx**2 + dy**2)  
            exp_alpha = np.arctan2(dy, dx) - self.particles[:, 2]  
  
            # Calculate weights using a Gaussian distribution  
            self.weights *= np.exp(-((d - exp_d)**2 / (2 * 0.2**2) + (alpha - exp_alpha)**2 / (2 * 0.2**2)))  
  
        # Normalize weights  
        self.weights += 1.e-300  # avoid round-off to zero  
        self.weights /= sum(self.weights)  
  
    def resample_particles(self):  
        indices = np.random.choice(self.num_particles, self.num_particles, p=self.weights)  
        self.particles = self.particles[indices]  
        self.weights = np.ones(self.num_particles) / self.num_particles  # Reset weights after resampling  
    
  
    def run_particle_filter(self):    
        self.visualize_landmarks()  # Visualize landmarks once at the beginning    
        particle_objects = []  # Store particle objects for visualization    
    
        estimated_states = []    
        for index, (odometry, landmark_obs) in enumerate(zip(self.odometry_data, self.landmark_data.reshape(-1, len(self.map_data) * 2))):    
            self.motion_update(odometry)    
            self.measurement_update(landmark_obs)    
            self.resample_particles()    
            estimated_state = np.average(self.particles, axis=0, weights=self.weights)    
            estimated_states.append(estimated_state)    
    
            #particle_objects = self.update_particle_visualization(particle_objects)  
    
        self.visualize_robot(estimated_states)    
        return np.array(estimated_states)  

    def visualize_robot(self, estimated_states):   
        simplified_trajectory = []  
        for state in estimated_states:  
            x, y, theta = state  
            simplified_trajectory.append([x, y, theta])  # Only include x, y, and theta  
    
        # Now pass this simplified trajectory to the CarRobot for visualization  
        self.car_robot.visualize_given_trajectory(simplified_trajectory)  
 
    def visualize_landmark_observations(self, estimated_state): 
        x, y, theta = estimated_state  
        for i, landmark in enumerate(self.map_data):  
            landmark_x, landmark_y = landmark[:2]  
            # Optionally, calculate expected observation from estimated_state to landmark  
            # For simplicity, just drawing a line from estimated position to landmark  
            self.viz_out.add_line([[x, y, 0.5], [landmark_x, landmark_y, 0.5]], color="0xff0000")  # Red lines  


    def visualize_landmarks(self):    
        for i, landmark in enumerate(self.map_data):    
            x, y = landmark[:2]  
            geom = sphere(f'landmark_{i}', 0.5, [x, y, 0.5], [1, 1, 0, 0])  
            self.viz_out.add_obstacle(geom, "0x0000ff")  
  

 
    def update_particle_visualization(self, particle_objects):  
        # Remove previous particle objects  
        for particle_object in particle_objects:  
            self.viz_out.remove_obstacle(particle_object)  
          
        # Clear the list of particle objects  
        particle_objects.clear()  
          
        # Add updated particle objects  
        for i, particle in enumerate(self.particles):  
            x, y, theta = particle  
            geom = sphere(f'particle_{i}', 0.05, [x, y, 0.5], [0.7, 0.7, 0.7, 0.5])  
            self.viz_out.add_obstacle(geom, "0x808080")  # Grey color for particles  
            particle_objects.append(geom)  
          
        return particle_objects

if __name__ == "__main__":  
    viz_out = threejs_group(js_dir="../js")  
    parser = argparse.ArgumentParser()  
    parser.add_argument("--map", type=str, required=True)  
    parser.add_argument("--odometry", type=str, required=True)  
    parser.add_argument("--landmarks", type=str, required=True)  
    parser.add_argument("--plan", type=str, required=True)  
    parser.add_argument("--num_particles", type=int, required=True)  
    args = parser.parse_args()  
  
    particle_filter = UnknownStartParticleFilter(viz_out, args.num_particles, args.map, args.odometry, args.landmarks, args.plan)  
    particle_filter_states = particle_filter.run_particle_filter()  
    viz_out.to_html("particle_filter_kidnapped.html", "../out/")  
