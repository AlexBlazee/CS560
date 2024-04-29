import numpy as np
import argparse
from geometry import * 
from threejs_group import *
import copy

class CarRobot:
    def __init__(self, q0 , actuation_noise, odometry_noise , observation_noise, viz_out , landmarks = None, L=1.5, dt=0.1 ):
        self.L = L
        self.dt = dt
        self.q0 = q0
        self.viz_out = viz_out
        self.actuation_noise = actuation_noise
        self.odometry_noise = odometry_noise
        self.observation_noise = observation_noise
        self.landmarks = landmarks

        self.actuation_noise_model = {'H': [0.3 , 0.2] , 'L' : [0.1 , 0.05] , 'None': None}  # v,phi
        self.odometry_noise_model = {'H': [0.15 , 0.1] , 'L' : [0.05 , 0.03] , 'None': None} # v,phi
        self.observation_noise_model = {'H': [0.5 , 0.25] , 'L' : [0.1 , 0.1] , 'None': None} # d, alpha


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
        theta_dot = (theta_dot + np.pi) % (2 * np.pi) - np.pi
        return np.array([x_dot, y_dot, theta_dot]),[v,phi]

    def odometry_measurement(self, u_e):
        v, phi = u_e
        if self.odometry_noise :
            if v!= 0:
                v = np.random.normal(v, np.sqrt(self.odometry_noise[0]))
            if  phi!= 0:
                phi = np.random.normal(phi, np.sqrt(self.odometry_noise[1]))
        return [v, phi]

    def landmark_observation(self, q, landmark):
        x, y, theta = q
        # Ground truth observation
        d = np.sqrt((landmark[0] - x)**2 + (landmark[1] - y)**2) # ground truth
        alpha = np.arctan2(landmark[1]-y,landmark[0]-x) - theta  # ground truth 
        # TODO:-> Is this np.arctan2(landmark[1],landmark[0])
        # Add noise
        if self.observation_noise:
            d = np.random.normal(d, np.sqrt(self.observation_noise[0]))
            alpha = np.random.normal(alpha, np.sqrt(self.observation_noise[1]))
        return [d, alpha]

    def get_all_landmark_observations(self , q):
        landmarks_info_at_q = []
        for landmark in self.landmarks:
            landmarks_info_at_q.extend(self.landmark_observation(q,landmark))
        return landmarks_info_at_q

    def simulate_trajectory(self, u, q0 , duration=10):
        q = q0
        trajectory = [q]
        actuation_control_list = []
        odometry_reading_list = []
        landmark_mesurement_list = []
        for _ in range(int(duration/self.dt)):
            q_dot,u_e = self.car_dynamics(q, u)
            q = q + q_dot * self.dt
            trajectory.append(q)
            if self.actuation_noise != None:
                actuation_control_list.append(u_e)
            if self.odometry_noise != None:
                odometry_reading_list.append(self.odometry_measurement(u_e))
            if self.observation_noise != None:
                landmark_mesurement_list.append(self.get_all_landmark_observations(q))
        return np.array(trajectory),actuation_control_list,odometry_reading_list , landmark_mesurement_list, q
    
    def visualize_trajectory(self, u , q0 ):
        trajectory,_,_,_,_ = self.simulate_trajectory(u , q0)
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

    def visualize_linepath_color(self, trajectory_path ,  color):  
        trajectory_path = copy.deepcopy(trajectory_path)
        trajectory_path[:,2] = 0.5
        self.viz_out.add_line(trajectory_path, color) 
        return self.viz_out 

    def visualize_landmark_observation(self, landmark_data , trajectory):
        gray = "#808080"
        landmark_count = int(len(landmark_data[0])/2)
        observation_per_step = [[] for _ in range(landmark_count)]
        for t, observation in enumerate(landmark_data):
            for j in range(0,landmark_count):
                rel_dist,rel_angle = observation[j*2],observation[(2*j)+1]
                pos_x,pos_y,pos_theta = trajectory[t]
                l_pos_x = pos_x + (rel_dist * np.cos(rel_angle + pos_theta))
                l_pos_y = pos_y + (rel_dist * np.sin(rel_angle + pos_theta))
                observation_per_step[j].append([t , [l_pos_x,l_pos_y, 0.5] , [1,0,0,0] , gray])
        for j in range(landmark_count):
            geom = sphere('lm_'+str(j) , 0.5 , observation_per_step[j][0][1], observation_per_step[j][0][2] )
            self.viz_out.add_animation(geom , observation_per_step[j])
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
    observation_noise_model = {'high': [0.5 , 0.25] , 'low' : [0.1 , 0.1] , 'None': None} # d, alpha

    q0 = np.array([0, 0, 0]) # initial configuration
    car_robot = CarRobot(q0 = q0 ,
                            actuation_noise= actuation_noise_model[NOISE_TYPE] ,
                            odometry_noise= odometry_noise_model[NOISE_TYPE] ,
                            observatoin_noise = observation_noise_model[NOISE_TYPE],
                            viz_out= viz_out )
    car_robot.visualize_trajectory(np.array(control_vec) , q0 )  
    # viz_out.to_html("forward_dynamic_car_path.html" , "out/")
    # viz_out.to_html("forward_dynamic_car_path_u_1_0.html" , "out/")
    # viz_out.to_html("forward_dynamic_car_path_u_1_1.html" , "out/")
    viz_out.to_html("forward_dynamic_car_path_u_n1_n1.html" , "out/")
