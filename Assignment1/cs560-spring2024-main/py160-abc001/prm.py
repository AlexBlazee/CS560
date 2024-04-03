import argparse
import math
import numpy as np
from geometry import * 
from threejs_group import *
from arm_2 import *
from nearest_neighbors import * 
from arm_2 import ModifiedRoboticArm
from graph import *
import heapq
import time

class PRM():
    def __init__(self , robot_type , start , goal, max_nodes , obstacles_file, viz_out) -> None:
        self.viz_out = viz_out
        self.robot_type = robot_type
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.obstacles = self.read_obstacles(obstacles_file)
        self.visualize_obstacles()
        self.graph = {}
        self.max_nodes = max_nodes
        self.k_nearest = 6
        self.mra = ModifiedRoboticArm(self.viz_out)
        self.graph = Graph(self.start , self.goal)
        self.nearest_neighbor = NearestNeighbour(self.viz_out)
        
    def read_obstacles(self, obstacle_file):
        obstacles = []
        with open(obstacle_file, 'r') as file:
            obstacles = [list(map(float, line.strip().split())) for line in file]
        return obstacles

    def visualize_obstacles(self):
        blue="0x0000ff"
        for i,[x,y,z,r] in enumerate(self.obstacles):
            geom = sphere('obs'+str(i) , r , [x,y,z] , [1,0,0,0] )
            self.viz_out.add_obstacle(geom , blue)
        return 

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
        
    def connect_initial_nodes(self , is_direct = True):
        node_list = [np.array(x) for x  in self.graph.get_nodes()]
        
        for node_1 in node_list:
            for node_2 in node_list:
                if (node_1 != node_2).all():
                    if is_direct == False:
                        if( ((node_1 == self.start).all() or (node_1 == self.goal).all()) and ((node_2 == self.start).all() or (node_2 == self.goal).all())):
                            continue
                    # print("Nodes", node_1 , node_2)
                    if self.is_valid_edge(node_1 , node_2):                    
                        distance = self.nearest_neighbor.get_config_distance(self.robot_type , node_1 , node_2)
                        self.graph.add_edge(node_1.tobytes() , node_2.tobytes() , distance)
        
        # self.graph.print_node_info()
        # print("----------------------------")

        return


    def build_graph(self , is_direct):
        print("Building graph .....")
        self.graph.add_node(self.start.tobytes())
        self.graph.add_node(self.goal.tobytes())

        for i in range(2, self.max_nodes):
            # sample the new configuration
            if i%100 == 0 :
                print(i)
            flag = True
            while(flag):
                new_sample_config = self.generate_random_config()
                # print(f"\n{new_sample_config} \n ")
                if self.is_valid_node(new_sample_config) and tuple(new_sample_config) not in self.graph.all_configs:
                    flag = False
                    break
            self.graph.add_node(new_sample_config.tobytes())
            # find the neighbourhood of new sample and add valid edges
            if (i == 6) :
                self.connect_initial_nodes(is_direct)
            if (i > 6):
                neighbor_info = self.nearest_neighbor.get_nearest_neighbors(self.robot_type , new_sample_config , self.graph.all_configs , self.k_nearest)
                for neigh_config,distance in neighbor_info:
                    if self.is_valid_edge(new_sample_config , neigh_config) and (new_sample_config != neigh_config).all():
                        self.graph.add_edge(new_sample_config.tobytes() , np.array(neigh_config).tobytes() , distance)

                # find nearest neighbour configs
        return

    def search_path(self , is_heuristic):
        print("Searching Path .....")
        visited = set()
        queue = [(0 , self.start.tobytes() , [])]
        
        while queue:
            cost , current_node , path = heapq.heappop(queue)
            if current_node in visited:
                continue
            visited.add(current_node)
            path.append(np.frombuffer(current_node))
            if current_node == self.goal.tobytes():
                return path
            
            for neighbor,distance in self.graph.get_all_neighbors(current_node):
                if neighbor not in visited:
                    if is_heuristic == True:
                        new_cost = cost + distance + self.graph.heuristic[neighbor]
                    else:
                        new_cost = cost + distance
                    heapq.heappush(queue , (new_cost , neighbor , path.copy()))
        print("Path couldn't be found")
        return None
    
    def visualize_path(self , final_path):
        # first add small spheres and edges 
            # code modified onto the mra.visualize_path function
        # second visualization
        if final_path:
            robot_path = None
            for i in range(1,len(final_path)):
                if i == 1 :
                    robot_path = np.linspace(final_path[i-1] , final_path[i] , num = 100)
                else:
                    robot_path = np.append(robot_path , np.linspace(final_path[i-1] , final_path[i] , num = 100) , axis= 0)
            return self.mra.visualize_arm_path(None , None , robot_path , final_path)

        return

if __name__ == "__main__":

    viz_out = threejs_group(js_dir="../js")
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", type= str , required=True, choices=["arm", "vehicle"])
    parser.add_argument("--start", type=float, nargs="+", required=True)
    parser.add_argument("--goal", type=float, nargs="+" , required=True)
    parser.add_argument("--map", type= str ,required=True)    
    args = parser.parse_args()

    robot_type = args.robot
    start_config = args.start
    goal_config = args.goal
    obstacles_file = args.map       
    max_nodes = 500
    num_iterations = 10
    time_per_iteration   = []
    for i in range(num_iterations):
        start_time = time.time()
        print("Current Iteration :", i)
        # print(robot_type , start_config , goal_config, obstacles_file)
        viz_out = threejs_group(js_dir="../js")
        # print(robot_type , start_config , goal_config, obstacles_file)
        prm = PRM(robot_type , start_config , goal_config, max_nodes , obstacles_file, viz_out)
        # print("PRM object instantiated")
        # print(prm.obstacles)

        is_direct = False  # can it move directly from the start state to goal state 
        prm.build_graph(is_direct)
        # print("Graph Nodes:")
        # prm.graph.print_node_info()
        final_path = prm.search_path(is_heuristic= True )

        if final_path:
            print(" Final Path :\n")
            for configuration in final_path:
                print(configuration)
        
        prm.visualize_path(final_path)
        viz_out.to_html(f"prm_arm_solution_iter_{i}.html", "out/")
        # viz_out.to_html("prm_arm_solution_2.html", "out/")
        end_time =time.time()
        time_per_iteration.append(end_time - start_time)
        print(f"The code took {end_time - start_time} seconds to execute.")

    print(time_per_iteration)