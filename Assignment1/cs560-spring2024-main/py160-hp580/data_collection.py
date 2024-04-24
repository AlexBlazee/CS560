import math
import numpy as np
from threejs_group import *
import argparse
from car_prop import CarRobot
from rrt import RRTPlanner
import os 

class DataCollection:
    def __init__(self,viz_out, start, goal, obstacles_file):
        self.viz_out = viz_out
        self.start = start
        self.goal = goal
        self.obstacles = self.read_obstacles(obstacles_file)
    
    def load_landmarks(self, landmarks_file):  
        with open(landmarks_file, 'r') as file:  
            landmarks = [list(map(float, line.strip().split())) for line in file]  
        return landmarks  


    def read_obstacles(self, obstacles_file):  
        # Read the obstacles data from the file and return it as a numpy array  
        obstacles = np.loadtxt(obstacles_file)  
        return obstacles  

    def write_data_to_files(self, problem , noise , planned_actions , trajectory_data , actuation_data , odometry_data , landmark_data):
        # planner returned actions
        data_directory = os.path.abspath("./data/")  
        if not os.path.exists(data_directory):  
            os.makedirs(data_directory)
        with open(os.path.join(data_directory, f"plan_{problem}_{noise}.txt"), "w") as plan_file:  
            plan_file.write(f"{self.start[0]} {self.start[1]} {self.start[2]}\n")
            for action in planned_actions :
                line = " ".join(map(str , action))
                plan_file.write( line + "\n")
        # executed ground truth trajectory aka real trajectory
        with open(os.path.join(data_directory, f"gt_{problem}_{noise}.txt"), "w") as gt_file:  
            for pos in trajectory_data:
                line = " ".join(map(str, pos))
                gt_file.write( line + "\n")
        # executed ground truth trajectory aka real trajectory
        with open(os.path.join(data_directory, f"odometry_{problem}_{noise}.txt"), "w") as od_file:  
            for odo in odometry_data:
                line = " ".join(map(str, odo))
                od_file.write( line + "\n")    
        with open(os.path.join(data_directory, f"landmarks_{problem}_{noise}.txt"), "w") as lm_file:  
            for lm in landmark_data:
                line = " ".join(map(str, lm))
                lm_file.write( line + "\n")            
        with open(os.path.join(data_directory, f"actuations_{problem}_{noise}.txt"), "w") as ac_file:  
            for exe_control in actuation_data:
                line = " ".join(map(str, exe_control))
                ac_file.write( line + "\n")       

if __name__ == "__main__":
    viz_out = threejs_group(js_dir="../js")
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, required=True)
    parser.add_argument("--problem" , nargs=1  , type= float , required= True )
    parser.add_argument("--noise" , type=str , required= True , choices=['H','L','None'])
    args = parser.parse_args()
        
    obstacles_file = "maps/map.txt" # may change based on the problem
    start = [5 , 25, 0.5] # may change based on the problem
    goal = [17, 15, 0.5] # may change based on the problem

    data_collection = DataCollection(viz_out, start, goal, obstacles_file)  


    actuation_noise_model = {'H': [0.3 , 0.2] , 'L' : [0.1 , 0.05] , 'None': None}  # v,phi  
    odometry_noise_model = {'H': [0.15 , 0.1] , 'L' : [0.05 , 0.03] , 'None': None} # v,phi  
    observation_noise_model = {'H': [0.5 , 0.25] , 'L' : [0.1 , 0.1] , 'None': None} # d, alpha  

    LANDMARK_FILE_NAME = 'landmark_0.txt'  
    land_mark_pos = data_collection.load_landmarks(LANDMARK_FILE_NAME) # load land_mark data

    for _ in range(2):        
        # planning 
        planner = RRTPlanner( viz_out ,start, goal, obstacles_file , None , None , None)
        tree_path = planner.rrt()
        path , actions = tree_path
        complete_final_path = planner.get_complete_trajectory(tree_path)
        
        # execution
        execution_car = CarRobot(q0 = start ,
                                actuation_noise= actuation_noise_model[args.noise] ,
                                odometry_noise= odometry_noise_model[args.noise] ,
                                observation_noise = observation_noise_model[args.noise],
                                viz_out= viz_out, 
                                landmarks= land_mark_pos )
        
        current_state = start
        ground_truth_trajectory = []
        odometry_data = []
        landmark_data = []
        actuation_data = []
        for action in actions:
            trajectory,actuation_info,odometry_info,landmark_info = execution_car.simulate_trajectory(u=action[0], q0=current_state, duration= action[1])
            ground_truth_trajectory.extend(trajectory)
            actuation_data.extend(actuation_info)
            odometry_data.extend(odometry_info)
            landmark_data.extend(landmark_info)

        data_collection.write_data_to_files( args.problem[0] , args.noise , actions , ground_truth_trajectory , actuation_data , odometry_data , landmark_data)


    
    