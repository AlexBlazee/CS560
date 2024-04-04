import argparse
import math
import numpy as np
from geometry import * 
from threejs_group import *
from arm_2 import *
import math
import random 

class NearestNeighbour():
    def __init__(self , viz_out) -> None:
        self.viz_out = viz_out
        return

    def input_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--robot", required=True, choices=["arm", "vehicle"])
        parser.add_argument("--target", type=float, nargs="+", required=True)
        parser.add_argument("-k", type=int, required=True)
        parser.add_argument("--configs", required=True)    
        args = parser.parse_args()

        with open(args.configs, "r") as f:
            configs = [list(map(float, line.strip().split())) for line in f]
        
        return args , configs


    def get_config_distance(self, robot ,config_1 , config_2):
        if robot == "arm":
            distance = np.linalg.norm(np.array(config_1) - np.array(config_2))
        if robot == "vehicle":
            x1, y1, z1, qw1, qx1, qy1, qz1 = config_1
            x2, y2, z2, qw2, qx2, qy2, qz2 = config_2
            position_distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
            orientation_distance = 2 * np.arccos(np.abs(qw1 * qw2 + qx1 * qx2 + qy1 * qy2 + qz1 * qz2))
            distance = position_distance + orientation_distance
        return distance

    def get_nearest_neighbors(self, robot, target, configs, k):
        distances = [(config, self.get_config_distance( robot , config, target)) for config in configs]
        sorted_distances = sorted(distances, key=lambda x: x[1])
        return sorted_distances[:k]
    
    def get_prm_star_nearest_neighbors( self , robot , target , configs , k , radius , sample_count , gamma , d):
        distances = [[config, self.get_config_distance( robot , config, target)] for config in configs]
        radius = gamma * math.pow( (math.log10(sample_count)/sample_count) , 1/d)
        sorted_distances = sorted(distances, key=lambda x: x[1])
        neighbor_list = [sublist for sublist in sorted_distances if sublist[1] <= radius]
        return neighbor_list

    def visualize_vehicle(self,configs):
       
        for i,config in enumerate(configs):
            pos = config[:3]
            quat = config[3:]
            cube = box("car" + str(i), 2,1,1, pos , quat)
            rand_color = str(hex(random.randrange(0,2**24)))
            self.viz_out.add_obstacle(cube, rand_color)
        return self.viz_out          

    def visulalize(self , nearest_configs , robot,   target ):
        list_of_configs = nearest_configs
        list_of_configs.append(target)
        if robot == "arm":
            modified_robotic_arm = ModifiedRoboticArm(self.viz_out)
            return modified_robotic_arm.visualize_arms(list_of_configs)
        if robot == "vehicle":
            return self.visualize_vehicle(list_of_configs)

if __name__ == "__main__":

    CREATE_CONFIG = False
    ARM_CONFIG_FILE_NAME = "configs.txt"
    # creating configs.txt file with configurations ONLY FOR ARMS
    if CREATE_CONFIG == True:
        NUM_RANDOM_CONFIGS = 10
        with open(ARM_CONFIG_FILE_NAME , "w" ) as file:
            for i in range(NUM_RANDOM_CONFIGS):
                confs = np.random.uniform(-np.pi, np.pi, 3)
                line = ' '.join(str(conf) for conf in confs) + '\n'
                file.write(line)
    else:
        print(f"Reading Data from {ARM_CONFIG_FILE_NAME}")

    viz_out = threejs_group(js_dir="../js")
    nearestNeighbour = NearestNeighbour(viz_out)
    args , configs =  nearestNeighbour.input_parser()
 
    nearest_neighbors = nearestNeighbour.get_nearest_neighbors(args.robot, args.target, configs, args.k)
    neighbors = []
    print(f"Nearest {args.k} neighbors for target {args.target}:")

    for neighbor,_ in nearest_neighbors:
        neighbors.append(neighbor)
        print(neighbor)
    print("==================")
    nearestNeighbour.visulalize(neighbors ,args.robot , args.target)
    # viz_out.to_html("nearest_neigbours_arm_configs_1.html", "out/") # 0 0 0
    # viz_out.to_html("nearest_neigbours_arm_configs_2.html", "out/") # -2.44 0.39 1.22
    # viz_out.to_html("nearest_neigbours_arm_configs_3.html", "out/") # 1.047 -1.57 0.78
    # viz_out.to_html("nearest_neigbours_arm_configs_4.html", "out/") # -1.22 2.09 1.57
    # viz_out.to_html("nearest_neigbours_arm_configs_5.html", "out/") # -0.39 -2.09 -2.09

#    viz_out.to_html("nearest_neigbours_cube_configs_1.html", "out/") # 0 0 0
    # viz_out.to_html("nearest_neigbours_cube_configs_2.html", "out/") # -2.44 0.39 1.22
    # viz_out.to_html("nearest_neigbours_cube_configs_3.html", "out/") # 1.047 -1.57 0.78
    # viz_out.to_html("nearest_neigbours_cube_configs_4.html", "out/") # -1.22 2.09 1.57
    viz_out.to_html("nearest_neigbours_cube_configs_5.html", "out/") # -0.39 -2.09 -2.09
# TODO:
# 1. Describe briefly your implementation in the report.Make sure that you reason correctly regarding the topology of the robot’s configuration space 
#        and define accordingly the distance metric between pairs of configurations
# 2. 