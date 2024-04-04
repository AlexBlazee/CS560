import numpy as np
import argparse
from geometry import * 
from threejs_group import *

class CarRobot:
    def __init__(self, q0 , viz_out ,  L=1.5, dt=0.1 ):
        self.L = L
        self.dt = dt
        self.q0 = q0
        self.viz_out = viz_out

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

    def car_dynamics(self, q, u):
        x, y, theta = q
        v, phi = u
        x_dot = v * np.cos(theta)
        y_dot = v * np.sin(theta)
        theta_dot = (v / self.L) * np.tan(phi)
        return np.array([x_dot, y_dot, theta_dot])

    def simulate_trajectory(self, u, q0 , duration=10):
        q = q0
        trajectory = [q]
        for _ in range(int(duration/self.dt)):
            q_dot = self.car_dynamics(q, u)
            q = q + q_dot * self.dt
            trajectory.append(q)
        return np.array(trajectory)
    
    def visualize_trajectory(self, u , q0):
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

    viz_out = threejs_group(js_dir="../js")
    parser = argparse.ArgumentParser()
    parser.add_argument("--u", type=float, nargs="+", required=True)
    args = parser.parse_args()
    control_vec = args.u
    print(control_vec)

    q0 = np.array([0, 0, 0]) # initial configuration
    car_robot = CarRobot(q0 = q0 , viz_out= viz_out)
    car_robot.visualize_trajectory(np.array(control_vec) , q0)  
    # viz_out.to_html("forward_dynamic_car_path.html" , "out/")

    # viz_out.to_html("forward_dynamic_car_path_u_1_0.html" , "out/")
    # viz_out.to_html("forward_dynamic_car_path_u_1_1.html" , "out/")
    viz_out.to_html("forward_dynamic_car_path_u_n1_n1.html" , "out/")
