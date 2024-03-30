import argparse
import math
import numpy as np
from geometry import * 
from threejs_group import *
from prm import *
import time

class PRM_STAR(PRM):
    def __init__(self, robot_type, start, goal, max_nodes , obstacles_file, viz_out) -> None:
        super().__init__(robot_type, start, goal, max_nodes, obstacles_file, viz_out)
        # if robot_type == 'arm':
        #     self.dim = 3
        # elif robot_type == 'vehicle':
        #     self.dim = 7
       

    def build_graph(self , is_direct):
        self.graph.add_node(self.start.tobytes())
        self.graph.add_node(self.goal.tobytes())

        for i in range(2, self.max_nodes):
            if i % 100 == 0:
                print(i)
            flag = True
            while flag:
                new_sample_config = self.generate_random_config()
                if self.is_valid_node(new_sample_config) and tuple(new_sample_config) not in self.graph.all_configs:
                    flag = False
                    break
            self.graph.add_node(new_sample_config.tobytes())

            if i == 6:
                self.connect_initial_nodes(is_direct)
            if i > 6 :
                # neighbor_info = self.nearest_neighbor.get_prm_star_nearest_neighbors(self.robot_type , new_sample_config , self.graph.all_configs , self.k_nearest , i , self.gamma , self.dim)
                neighbor_info = self.nearest_neighbor.get_nearest_neighbors(self.robot_type , new_sample_config , self.graph.all_configs , int(math.log(i)))
                for neigh_config,distance in neighbor_info:
                    if self.is_valid_edge(new_sample_config , neigh_config) and (new_sample_config != neigh_config).all():
                        self.graph.add_edge(new_sample_config.tobytes() , np.array(neigh_config).tobytes() , distance)
        return

if __name__ == "__main__":
    start_time = time.time()
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
    max_nodes = 5000
    # print(robot_type , start_config , goal_config, obstacles_file)
    prm_star = PRM_STAR(robot_type , start_config , goal_config,  max_nodes , obstacles_file, viz_out)
    # print("PRM Star object instantiated")
    # print(prm_star.obstacles)
    is_direct = False  # can it move directly from the start state to goal state 
    prm_star.build_graph(is_direct)
    # print("Graph Nodes:")
    # prm_star.graph.print_node_info()
    final_path = prm_star.search_path(is_heuristic= True) # A* implementation

    if final_path:
        print(" Final Path :\n")
        for configuration in final_path:
            print(configuration)

    
    prm_star.visualize_path(final_path)
    # viz_out.to_html("prm_star_arm_solution_1.html", "out/")
    viz_out.to_html("prm_star_arm_solution_2.html", "out/")
    end_time =time.time()
    print(f"The code took {end_time - start_time} seconds to execute.")