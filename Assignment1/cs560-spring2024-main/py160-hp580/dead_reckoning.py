import argparse  
import numpy as np  
from data_collection import DataCollection  
from geometry import * 
import numpy as np
from geometry import * 
from threejs_group import *
import os  
from car_prop import CarRobot
import copy

class DeadReckoning:
    def __init__(self,viz_out , map_file, odometry_file, landmarks_file, plan_file ):
        self.viz_out = viz_out
        self.dead_reckoning_car = CarRobot(q0 = None ,actuation_noise= None, odometry_noise= None ,observation_noise = None,
                             viz_out= viz_out, landmarks= None )
        self.L = self.dead_reckoning_car.L
        self.start_state = self.load_start_state(plan_file)
        self.odometry_data = self.load_data(odometry_file)
        self.landmark_data = self.load_data(landmarks_file)
        self.landmarks =   self.load_landmarks(map_file)
       
        
    def load_landmarks(self, landmarks_file):  
        with open(landmarks_file, 'r') as file:  
            landmarks = [list(map(float, line.strip().split())) for line in file]  
        return landmarks   

    def visualize_landmarks(self):
        blue="0x0000ff"
        for i,[x,y] in enumerate(self.landmarks):
            geom = sphere('obs'+str(i) , 0.5 , [x,y,0.5] , [1,0,0,0] )
            self.viz_out.add_obstacle(geom , blue)
        return 

    def load_data(self, filename):  
        data = []  
        with open(filename, 'r') as file:  
            for line in file:  
                line = line.strip().replace('(', '').replace(')', '').replace(',', '').split()  
                data.append([float(x) for x in line])  
        return np.array(data)  

    def load_start_state(self, filename):
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip().split()
                line = [float(x) for x in line]
                break
        return line    

    # def visualize_given_trajectory_name(self, trajectory , name):
    #     yellow = "0xFFFF00"
    #     car_trajectory = []
    #     for t in range(len(trajectory)):
    #         x,y,theta = trajectory[t]
    #         quat = self.dead_reckoning_car.rotate_z_quaternion(theta=theta) 
    #         car_trajectory.append([t, [x,y,0.5] , quat , yellow ])
        
    #     cube = box(name, 2,1,1, car_trajectory[0][1], car_trajectory[0][2])
    #     self.viz_out.add_animation(cube , car_trajectory)
    #     return self.viz_out    

    # def visualize_landmark_observation(self, landmark_data , trajectory):
    #     gray = "#808080"
    #     landmark_count = int(len(landmark_data[0])/2)
    #     observation_per_step = [[] for _ in range(landmark_count)]
    #     for t, observation in enumerate(landmark_data):
    #         for j in range(0,landmark_count):
    #             rel_dist,rel_angle = observation[j*2],observation[(2*j)+1]
    #             pos_x,pos_y,pos_theta = trajectory[t]
    #             l_pos_x = pos_x + (rel_dist * np.cos(rel_angle + pos_theta))
    #             l_pos_y = pos_y + (rel_dist * np.sin(rel_angle + pos_theta))
    #             observation_per_step[j].append([t , [l_pos_x,l_pos_y, 0.5] , [1,0,0,0] , gray])

    #     for j in range(landmark_count):
    #         geom = sphere('lm_'+str(j) , 0.5 , observation_per_step[j][0][1], observation_per_step[j][0][2] )
    #         self.viz_out.add_animation(geom , observation_per_step[j])
    #     return self.viz_out

    def dead_reckoning_trajectory(self, odometry_data, start):  
        estimated_poses = [start]  
        current_pose = np.array(start)  
        for odo in odometry_data:  
            v, phi = odo  
            x, y, theta = current_pose 
            x_dot = v * np.cos(theta)
            y_dot = v * np.sin(theta) 
            theta_dot = (v / self.L) * np.tan(phi)
            theta_dot = (theta_dot + np.pi) % (2 * np.pi) - np.pi
            current_pose += np.array([x_dot , y_dot , theta_dot]) * 0.1
            estimated_poses.append(list(current_pose))  
        return np.array(estimated_poses)  
    
    def visualize_dead_reckoning(self, ground_truth, dead_reckoning ):  
        green = "0x00ff00"  
        red = "0xff0000"  
        ground_truth = copy.deepcopy(ground_truth)
        dead_reckoning = copy.deepcopy(dead_reckoning)
        ground_truth[:,2] = 0.5
        dead_reckoning[:,2] = 0.5
        self.viz_out.add_line(ground_truth, green) 
        self.viz_out.add_line(dead_reckoning , red)
        return self.viz_out
  
if __name__ == "__main__":  
    viz_out = threejs_group(js_dir="../js")  
    parser = argparse.ArgumentParser()  
    parser.add_argument("--map", type=str, required=True)  
    parser.add_argument("--odometry", type=str, required=True)  
    parser.add_argument("--landmarks", type=str, required=True)  
    parser.add_argument("--plan", type=str, required=True)  
    args = parser.parse_args()  

    green = "0x00ff00"  
    red = "0xff0000"  

    dead_reckoning = DeadReckoning(viz_out , args.map, args.odometry, args.landmarks, args.plan)
    dead_reckoning.visualize_landmarks()
    dead_reckoning_states = dead_reckoning.dead_reckoning_trajectory(dead_reckoning.odometry_data , dead_reckoning.start_state)
    ground_truth_data = dead_reckoning.load_data(f"./py160-hp580/data/gt_{args.odometry.split('_')[-2]}_{args.odometry.split('_')[-1].split('.')[0]}.txt")  
    problem_id = args.odometry.split('_')[-2]


    dead_reckoning.dead_reckoning_car.visualize_linepath_color( ground_truth_data, green )
    dead_reckoning.dead_reckoning_car.visualize_linepath_color( dead_reckoning_states, red )      
    # dead_reckoning.visualize_dead_reckoning( ground_truth_data, dead_reckoning_states )  
    dead_reckoning.dead_reckoning_car.visualize_given_trajectory_name(dead_reckoning_states , "dead_reckon")
    dead_reckoning.dead_reckoning_car.visualize_given_trajectory_name(ground_truth_data , "ground_truth")    
    dead_reckoning.dead_reckoning_car.visualize_landmark_observation(dead_reckoning.landmark_data ,dead_reckoning_states )
    viz_out.to_html(f"dead_reckoning_{int(problem_id)}.html", "out/")  
 
    # #python dead_reckoning.py --map maps/map.txt --odometry data/odometry_1.0_L.txt --landmarks data/landmarks_1.0_L.txt --plan data/plan_1.0_L.txt