import argparse
import math
import numpy as np
from geometry import * 
from threejs_group import *
from arm_2 import *
from nearest_neighbors import * 
from arm_2 import ModifiedRoboticArm
from graph import *

class PRM():
    def __init__(self , robot_type , start , goal, obstacles_file, viz_out) -> None:
        self.viz_out = viz_out
        self.robot_type = robot_type
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = self.read_obstacles(obstacles_file)
        self.graph = {}
        self.max_nodes = 5000
        self.k_nearest = 6
        self.mra = ModifiedRoboticArm(self.viz_out)
        self.graph = Graph()
        
    def input_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--robot", type= str , required=True, choices=["arm", "vehicle"])
        parser.add_argument("--start", type=float, nargs="+", required=True)
        parser.add_argument("--goal", type=float, nargs="+" , required=True)
        parser.add_argument("--map", type= str ,required=True)    
        args = parser.parse_args()
        return args

    def read_obstacles(self, obstacle_file):
        obstacles = []
        with open(obstacle_file, 'r') as file:
            obstacles = [list(map(float, line.strip().split())) for line in file]
        return obstacles

    def generate_random_config(self):
        if self.robot_type == 'arm':
            return np.random.uniform(-np.pi, np.pi, 3)
        elif self.robot_type == 'vehicle':
            # TODO : implement vehicle initialization
            print(" This part is not initialized yet")
            return -1

    def is_valid_edge(self, config1, config2):        
        path = self.mra.calculate_arm_path_without_collision(config1 , config2 , self.obstacles)
        if path == -1:
            return False 
        return path,True
    
    def is_valid_node(self , config ):
        if self.mra.aabb_env_obstacle_collision_check(config , self.obstacles) == False:
            return True # no collision
        else:
            return False

    def build_graph(self):
        self.graph.add_node(self.start.tobytes())
        self.graph.add_node(self.goal.tobytes())
        

        for i in range(2, self.max_nodes):
            # sample the new configuration
            flag = True
            while(flag):
                new_sample_config = self.generate_random_config()
                if self.is_valid_node(new_sample_config):
                    flag = False
                if(flag == False):
                    break

            self.graph.add_node(new_sample_config.to_bytes())
            # find the neighbourhood of new sample and add valid edges
            if (i >= 6) :
                # find nearest neighbour configs
                


        
            

            
        

    def search_path(self):



if __name__ == "__main__":
    viz_out = threejs_group(js_dir="../js")
    prm = PRM()
    args = prm.input_parser()

    robot_type = args.robot
    start_config = args.start
    goal_config = args.goal
    obstacles_file = args.map   