import argparse  
import numpy as np  
from data_collection import DataCollection  
from geometry import * 
import numpy as np
from geometry import * 
from threejs_group import *
import os  
from car_prop import CarRobot

class DeadReckoning:
    def __init__(self,viz_out):
        self.viz_out = viz_out
        self.dead_reckoning_car = CarRobot(q0 = None ,actuation_noise= None, odometry_noise= None ,observation_noise = None,
                             viz_out= viz_out, landmarks= None )
        self.L = self.dead_reckoning_car.L

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

    def dead_reckoning_trajectory(self, odometry_data, start):  
        estimated_poses = [start]  
        current_pose = np.array(start)  
    
        for odo in odometry_data:  
            v, phi = odo  
            x, y, theta = current_pose 
            x_dot = v * np.cos(theta)
            y_dot = v * np.sin(theta) 
            theta_dot = (v / self.L) * np.tan(phi)
            current_pose += np.array([x_dot , y_dot , theta_dot]) * 0.1
            # theta_new = theta + v * phi * 0.1  #???
            # x_new = x + v * np.cos(theta) * 0.1  
            # y_new = y + v * np.sin(theta) * 0.1  
            # current_pose = np.array([x_new, y_new, theta_new])  
            estimated_poses.append(current_pose)  
    
        return np.array(estimated_poses)  
    
    def visualize_dead_reckoning(self, ground_truth, dead_reckoning):  
        green = "0x00ff00"  
        red = "0xff0000"  
        ground_truth[:,2] = 0.5
        dead_reckoning[:,2] = 0.5
        self.viz_out.add_line(ground_truth, green) 
        self.viz_out.add_line(dead_reckoning , red)
  
if __name__ == "__main__":  
    viz_out = threejs_group(js_dir="../js")  
    parser = argparse.ArgumentParser()  
    parser.add_argument("--map", type=str, required=True)  
    parser.add_argument("--odometry", type=str, required=True)  
    parser.add_argument("--landmarks", type=str, required=True)  
    parser.add_argument("--plan", type=str, required=True)  
    args = parser.parse_args()  

    dead_reckoning = DeadReckoning(viz_out)
    start_state = dead_reckoning.load_start_state(args.plan)  
    odometry_data = dead_reckoning.load_data(args.odometry)  
    ground_truth_data = dead_reckoning.load_data(f"./py160-hp580/data/gt_{args.odometry.split('_')[-2]}_{args.odometry.split('_')[-1].split('.')[0]}.txt")  
    dead_reckoning_poses = dead_reckoning.dead_reckoning_trajectory(odometry_data, start_state)  


    dead_reckoning.visualize_dead_reckoning( ground_truth_data, dead_reckoning_poses)  
    viz_out.to_html("dead_reckoning.html", "out/")  
 
    # #python dead_reckoning.py --map maps/map.txt --odometry data/odometry_1.0_L.txt --landmarks data/landmarks_1.0_L.txt --plan data/plan_1.0_L.txt