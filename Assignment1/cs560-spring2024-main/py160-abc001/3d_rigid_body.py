'''
def interpolate_rigid_body(start_position, goal_position):
def visualize_robot_path(path):
def A_star(start, goal, scene):

'''

from geometry import * 
from threejs_group import *
import random
import numpy as np
import json
from create_scene import Scene
from collision_checking import CollisionCheck
import heapq
import math
import numpy as np

class Cell:
    def __init__(self) -> None:
        self.parent = [0 , 0 , 0]
        self.f = float('inf')
        self.g = float('inf')
        self.h = 0

class ThreeDRigidBodyMotion():
    def __init__(self , viz_out , Grid_Rows , Grid_Cols , Grid_Depth , scene_dict) -> None:
        self.Grid_Rows = Grid_Rows
        self.Grid_Cols = Grid_Cols
        self.Grid_Depth = Grid_Depth
        self.scene_dict = scene_dict
        self.viz_out = viz_out
        return

    def check_cell_collision(self ,  row_id , col_id , dep_id):
        # use the collision checker to build a map of the grid
        collision = CollisionCheck(None)
        cube = box("box_temp", 5,5,5, [row_id , col_id , dep_id], [1,0,0,0])
        # check collision with evry sphere . change color , change and change state
        for i in self.scene_dict:
            n_s,r_s,p_s,o_s,_ = list(self.scene_dict[i].values())
            sp_geom = sphere(n_s , r_s , np.array(p_s) + 50 , o_s ) # 50 is added to accomadate for the row,cold,dep being positive
            if collision.collision_check( cube , sp_geom): 
                return True
        return False

    def interpolate_rigid_body(self , start_position , goal_position):
        step = (np.array(goal_position) - np.array(start_position))/5
        for i in range(5):
            int_x,int_y,int_z  = start_position + step
            if self.check_cell_collision(int_x , int_y , int_z):
                return True
        return False 

    def A_star( self, init_state , goal_state , scene_dict ):

        def check_cell_validity( row_id , col_id , dep_id):
            
            return row_id >= 0 and row_id < self.Grid_Rows and col_id >= 0 and col_id < self.Grid_Cols  and dep_id >= 0 and dep_id < self.Grid_Depth 

        def check_is_goal_state( row_id , col_id , dep_id , goal_state):
            return np.all(np.array(goal_state) == np.array([row_id , col_id , dep_id])) 
            #return goal_state[0] == row_id and goal_state[1] == col_id and goal_state[2] == dep_id

        def get_euclidian_heuristic_val (row_id , col_id , dep_id , goal_state):
            return math.sqrt(sum((goal_state - np.array([row_id , col_id  , dep_id])) ** 2))

        def get_path( cell_props , goal_state):
            path = []
            row_id,col_id,dep_id = goal_state
            while not (cell_props[row_id , col_id , dep_id].parent == [row_id , col_id , dep_id] ):
                path.append((row_id,col_id,dep_id))
                row_id,col_id,dep_id = cell_props[row_id , col_id , dep_id].parent

            path.append((row_id, col_id , dep_id))
            path.reverse()
            return path
       
        # Initial states and Goal state are valid
        if not check_cell_validity(init_state[0], init_state[1] , init_state[2]) or not check_cell_validity(goal_state[0], goal_state[1] , goal_state[2]):
            print("Initial State or Goal State is invalid")
            return
    
        # Check if the source and destination are unblocked
        if  self.check_cell_collision(init_state[0], init_state[1] , init_state[2]) or  self.check_cell_collision( goal_state[0], goal_state[1] , goal_state[2]):
            print("Initial State or Goal State is in Collision")
            return
    
        # Check if we are already at the destination
        if check_is_goal_state(init_state[0], init_state[1] , init_state[2] , goal_state):
            print("Already at Goal State")
            return

        closed_list = np.zeros((self.Grid_Rows,self.Grid_Cols,self.Grid_Depth)) > 1 # False array
        cell_props = np.array([[[Cell() for _ in range(self.Grid_Depth)] for _ in range(self.Grid_Cols)] for _ in range(self.Grid_Rows)])

        i,j,k = init_state
        cell_props[i,j,k].f = 0
        cell_props[i,j,k].g = 0
        cell_props[i,j,k].h = 0
        cell_props[i,j,k].parent = [i,j,k]
        
        open_list = []
        heapq.heappush(open_list , (0.0 , i, j, k ))

        at_goal_state = False

        while len(open_list) >0:
            p = heapq.heappop(open_list)
            _,i,j,k = p
            closed_list[i,j,k] = True

            directions = [[5,0,0],[-5,0,0],[0,5,0],[0,-5,0],[0,0,5],[0,0,-5]]
            for des in directions:
                new_i, new_j , new_k = i + des[0] , j + des[1] , k + des[2]
                if check_cell_validity(new_i,new_j,new_k) and not self.check_cell_collision( new_i , new_j , new_k) and not closed_list[new_i,new_j,new_k] and not self.interpolate_rigid_body([i,j,k],[new_i,new_j,new_k]):
                    if check_is_goal_state(new_i , new_j , new_k , goal_state ):
                        cell_props[new_i , new_j , new_k].parent = [i,j,k]
                        # goal state reached
                        final_path = get_path(cell_props , goal_state)
                        at_goal_state = True
                        return final_path
                    else:
                        g_new = cell_props[i,j,k].g + 5.0
                        h_new = get_euclidian_heuristic_val(new_i , new_j , new_k , goal_state)
                        f_new = g_new + h_new
                        if cell_props[new_i,new_j,new_k].f == float('inf') or cell_props[new_i,new_j,new_k].f > f_new :
                            heapq.heappush(open_list , (f_new , new_i , new_j , new_k))
                            cell_props[new_i,new_j,new_k].f = f_new
                            cell_props[new_i,new_j,new_k].g = g_new
                            cell_props[new_i,new_j,new_k].h = h_new
                            cell_props[new_i,new_j,new_k].parent = [i,j,k]
        
        if not at_goal_state:
            print(" Path could not be found")

    def visualize_robot_path(self, path):
        blue="0x0000ff"
        self.viz_out = Scene(self.viz_out).visualize_scene(self.scene_dict)
        cube_trajectory = []
        line_trajectory = []
        
        for t in range(len(path)):   
            coord = np.array(path[t]) - 50  
            cube_trajectory.append([t, coord , [1,0,0,0] , blue ])
            line_trajectory.append(coord)

        cube = box("box0", 5,5,5, cube_trajectory[0][1], cube_trajectory[0][2])
        self.viz_out.add_animation(cube , cube_trajectory)
        self.viz_out.add_line(line_trajectory , blue)
        return self.viz_out
        # viz_out.add_animation for all the spheres 


if __name__ == "__main__":  

    file_names = ["scene_1.txt" , "scene_2.txt"  ,"scene_3.txt"  ,"scene_4.txt"  ,"scene_5.txt" ]

    for file in file_names:
        
        viz_out = threejs_group(js_dir="../js")
        scene_dict = Scene(viz_out).scene_from_file(file)

        Grid_Rows , Grid_Cols , Grid_Depth = 100, 100, 100
        rigid_body_motion = ThreeDRigidBodyMotion(viz_out ,Grid_Rows , Grid_Cols , Grid_Depth , scene_dict)
        Grid = np.ones((Grid_Rows , Grid_Cols , Grid_Depth))

        # taking positive values but collsion is accomodated
        # will convert the above path to the new path
        start_flag = True        
        goal_flag = True
        while(start_flag):
            start_pos = [random.randint(0,19) * 5 , random.randint(0,19) * 5 ,random.randint(0,19) * 5]
            if not rigid_body_motion.check_cell_collision(start_pos[0] , start_pos[1] , start_pos[2]):
                start_flag = False
        
        while(goal_flag):
            goal_pos = [random.randint(0,19) * 5 , random.randint(0,19) * 5 ,random.randint(0,19) * 5]
            if not rigid_body_motion.check_cell_collision(goal_pos[0] , goal_pos[1] , goal_pos[2]):
                goal_flag = False

        print(f"start position : {start_pos} and goal position : {goal_pos}")
        final_path = rigid_body_motion.A_star(start_pos , goal_pos , scene_dict)
        print(f" The final path is :\n {final_path} ")
        rigid_body_motion.visualize_robot_path(final_path)
        animation_file_name = file[:-4]+ "a_start_path_animation.html"
        viz_out.to_html(animation_file_name, "out/")
        




        

