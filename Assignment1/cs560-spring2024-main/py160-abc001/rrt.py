import math
import numpy as np
from geometry import * 
from threejs_group import *
from nearest_neighbors import * 
from graph import *
import argparse
from scipy.spatial.distance import euclidean



class RRTPlanner:
    def __init__(self, viz_out,  start, goal, obstacles_file):
        self.viz_out = viz_out
        self.start = start
        self.goal = goal
        self.obstacles = self.read_obstacles(obstacles_file)
        self.tree = [start]
        self.dt = 0.1
        self.L = 1.5
        self.goal_threshold_trans = 0.1
        self.goal_threshold_rot = 0.5

    def read_obstacles(self, obstacle_file):
        obstacles = []
        with open(obstacle_file, 'r') as file:
            obstacles = [list(map(float, line.strip().split())) for line in file]
        return obstacles

    def sample(self):
        x = np.random.uniform(-50,50)
        y = np.random.uniform(-50,50)
        theta = np.random.uniform(-np.pi, np.pi)
        return np.array([x, y, theta])
    
    def nearest(self, tree, state):
        distances = [euclidean(state, node) for node in tree]
        return tree[np.argmin(distances)]
    
    def choose_control(self, near, rand):
        dx = rand[0] - near[0]
        dy = rand[1] - near[1]
        dtheta = rand[2] - near[2]
        v = np.sqrt(dx**2 + dy**2)
        phi = np.arctan2(dy, dx) - near[2]
        return np.array([v, phi])
    
    def is_collision_free_check(self, state , sphere_position , sphere_radius , car_position):
        r = sphere_radius
        s_x,s_y,s_z = np.array(sphere_position)
        lx,ly,lz = np.array([2,1,1])/2
        local_vertices = np.array([[-lx,-ly,-ly],[lx,-ly,-lz],[-lx,ly,-lz],[lx,ly,-lz],[-lx,-ly,lz],[lx,-ly,lz],[-lx,ly,lz],[lx,ly,lz]])
        rotation_matrix = ################ self.transformer.quaternion_to_matrix(orientation)
        transformed_points = ############## np.dot(rotation_matrix, local_vertices.T)
        x_min,y_min,z_min = np.min(transformed_points , axis = 1) + car_position
        x_max,y_max,z_max = np.max(transformed_points , axis = 1) + car_position
        # get box closest point to sphere center by clamping
        X = max(x_min , min(s_x , x_max))
        Y = max(y_min , min(s_y , y_max))
        Z = max(z_min , min(s_z , z_max))
        distance  = np.sqrt((X - s_x)*(X - s_x)  + (Y - s_y)*(Y - s_y) + (Z - s_z)*(Z - s_z))

        if distance > r :
            return False
        else:
            return True
        
    def obstacle_collision_check(self , configuration , obstacles_list , state , sphere_position ,sphere_radius , car_position ):
        for obstacle in obstacles_list:
            #x,y,z,r = obstacle
            if self.is_collision_free_check( state , sphere_position , sphere_radius , car_position):
                return True
        return False    

    def calculate_car_path_without_collision(self, start_configuration, end_configuration, obstacles_list, steps=100):
        path = []
        for t in np.linspace(0, 1, steps):
            interpolated_configuration =  np.array(start_configuration) + t * (np.array(end_configuration) - np.array(start_configuration))
            if self.obstacle_collision_check(interpolated_configuration , obstacles_list) == True:
                return -1
            path.append(tuple(interpolated_configuration))
        return path    


    def rrt(self, max_iterations=1000):
        """
        Implement the RRT algorithm to find a collision-free trajectory from start to goal.
        """
        for _ in range(max_iterations):
            rand_state = self.sample()
            nearest_state = self.nearest(self.tree, rand_state)
            control = self.choose_control(nearest_state, rand_state)
            new_state = self.simulate(nearest_state, control)

            if self.is_collision_free(new_state):
                self.tree.append(new_state)
                if np.linalg.norm(new_state[:2] - self.goal[:2]) < self.goal_threshold_trans and \
                   np.abs(new_state[2] - self.goal[2]) < self.goal_threshold_rot:
                    return self.tree
        return None


if __name__ == "__main__":
    viz_out = threejs_group(js_dir="../js")
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", nargs=3, type=float, required=True)
    parser.add_argument("--goal", nargs=3, type=float, required=True)
    parser.add_argument("--map", type=str, required=True)
    args = parser.parse_args()
    
    start = np.array(args.start)
    goal = np.array(args.goal)
    obstacles_file = args.map   
    
    planner = RRTPlanner( viz_out ,start, goal, obstacles_file)
    tree = planner.rrt()

    if tree is not None:
        planner.visualize_rrt()
        planner.visualize_trajectory(tree)
    else:
        print("No path found.")