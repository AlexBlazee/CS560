import numpy as np
import argparse
from geometry import * 
from threejs_group import *

class CarRobot:
    def __init__(self, q0 , actuation_noise, odometry_noise , observation_noise, viz_out ,  L=1.5, dt=0.1 ):
        self.L = L
        self.dt = dt
        self.q0 = q0
        self.viz_out = viz_out
        self.actuation_noise = actuation_noise
        self.odometry_noise = odometry_noise
        self.observation_noise = observation_noise

    @staticmethod
    def rotate_z_quaternion(theta):
        return [np.cos(theta/2), 0, 0, np.sin(theta/2)]

    @staticmethod
    def quaternion_to_matrix(q):
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]])

    def car_dynamics(self, q, u ):
        x, y, theta = q
        v, phi = u
        if self.actuation_noise:
            v = np.random.normal(v , np.sqrt(self.actuation_noise[0])) # actuation_noise[0] has variane of velocity noise
            phi = np.random.normal(phi , np.sqrt(self.actuation_noise[1])) # actuation_noise[1] has variance of phi noise
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = (v / self.L) * np.tan(phi)
        return np.array([x_dot, y_dot, theta_dot])

    def odometry_measurement(self, u_e):
        v, phi = u_e
        if self.odometry_noise :
            if v!= 0:
                v = np.random.normal(v, np.sqrt(self.odometry_noise[0]))
            if  phi!= 0:
                phi = np.random.normal(phi, np.sqrt(self.odometry_noise[1]))
        return np.array([v, phi])


    def landmark_observation(self, q, landmark):
        x, y, theta = q
        # Ground truth observation
               
        d = np.sqrt((landmark[0] - x)**2 + (landmark[1] - y)**2) # ground truth
        alpha = np.arctan2(landmark[1],landmark[0]) - theta  # ground truth
        # Add noise
        if self.observation_noise:
            d = np.random.normal(d, np.sqrt(self.observation_noise[0]))
            alpha = np.random.normal(alpha, np.sqrt(self.observation_noise[1]))
        return np.array([d, alpha])

    def simulate_trajectory(self, u, q0 , duration=10):
        q = q0
        trajectory = [q]
        for _ in range(int(duration/self.dt)):
            q_dot = self.car_dynamics(q, u)
            q = q + q_dot * self.dt
            trajectory.append(q)
        return np.array(trajectory)
    
    def visualize_trajectory(self, u , q0 ):
        trajectory = self.simulate_trajectory(u , q0)
        yellow = "0xFFFF00"
        black = "0x000000"
        car_trajectory = []
        line_trajectory = []
        
        for t in range(len(trajectory)):
            x,y,theta = trajectory[t]
            quat = self.rotate_z_quaternion(theta=theta) 
            car_trajectory.append([t, [x,y,0.5] , quat , yellow ])
            line_trajectory.append([x,y,0.5])

        cube = box("car", 2,1,1, car_trajectory[0][1], car_trajectory[0][2])
        self.viz_out.add_animation(cube , car_trajectory)
        self.viz_out.add_line(line_trajectory , black)
        return self.viz_out
    
    def visualize_given_trajectory(self, trajectory ):
        yellow = "0xFFFF00"
        black = "0x000000"
        car_trajectory = []
        line_trajectory = []
        
        for t in range(len(trajectory)):
            x,y,theta = trajectory[t]
            quat = self.rotate_z_quaternion(theta=theta) 
            car_trajectory.append([t, [x,y,0.5] , quat , yellow ])
            line_trajectory.append([x,y,0.5])

        cube = box("car", 2,1,1, car_trajectory[0][1], car_trajectory[0][2])
        self.viz_out.add_animation(cube , car_trajectory)
        self.viz_out.add_line(line_trajectory , black)
        return self.viz_out    

if __name__ == "__main__":

    NOISE_TYPE = 'low' # 'None', 'low' , 'high'
    viz_out = threejs_group(js_dir="../js")
    parser = argparse.ArgumentParser()
    parser.add_argument("--u", type=float, nargs="+", required=True)
    args = parser.parse_args()
    control_vec = args.u
    print(control_vec)
    actuation_noise_model = {'high': [0.3 , 0.2] , 'low' : [0.1 , 0.05] , 'None': None}  # v,phi
    odometry_noise_model = {'high': [0.15 , 0.1] , 'low' : [0.05 , 0.03] , 'None': None} # v,phi
    observation_noide_model = {'high': [0.5 , 0.25] , 'low' : [0.1 , 0.1] , 'None': None} # d, alpha

    q0 = np.array([0, 0, 0]) # initial configuration
    car_robot = CarRobot(q0 = q0 ,
                            actuation_noise= actuation_noise_model[NOISE_TYPE] ,
                            odometry_noise= odometry_noise_model[NOISE_TYPE] ,
                            observatoin_noise = observation_noide_model[NOISE_TYPE],
                            viz_out= viz_out )
    car_robot.visualize_trajectory(np.array(control_vec) , q0 )  
    # viz_out.to_html("forward_dynamic_car_path.html" , "out/")
    # viz_out.to_html("forward_dynamic_car_path_u_1_0.html" , "out/")
    # viz_out.to_html("forward_dynamic_car_path_u_1_1.html" , "out/")
    viz_out.to_html("forward_dynamic_car_path_u_n1_n1.html" , "out/")
