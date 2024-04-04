import math
import numpy as np
from geometry import * 
from threejs_group import *
from nearest_neighbors import * 
import argparse
from scipy.spatial.distance import euclidean
from car_prop import CarRobot
from tree import Tree,TreeNode

class RRTPlanner:
    def __init__(self, viz_out,  start, goal, obstacles_file):
        self.viz_out = viz_out
        self.start = start
        self.goal = goal
        self.obstacles = self.read_obstacles(obstacles_file)
        self.visualize_obstacles()
        self.tree = Tree(tuple(start))
        self.goal_threshold_trans = 0.1
        self.goal_threshold_rot = 0.5
        self.car_robot = CarRobot( self.start , self.viz_out)

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

    def visualize_obstacles(self):
        blue="0x0000ff"
        for i,[x,y,z,r] in enumerate(self.obstacles):
            geom = sphere('obs'+str(i) , r , [x,y,z] , [1,0,0,0] )
            self.viz_out.add_obstacle(geom , blue)
        return 

    def nearest(self, tree_nodes, state): ## update the function
        distances = [euclidean(state, node) for node in tree_nodes]
        return tree_nodes[np.argmin(distances)]
    
    def choose_guided_control(self, near, rand):
        dx = rand[0] - near[0]
        dy = rand[1] - near[1]
        dtheta = rand[2] - near[2]
        v = int(np.sqrt(dx**2 + dy**2))%5
        phi = np.arctan2(dy, dx) - near[2]
        if phi >1: phi = 1
        elif phi < -1 : phi = -1
        # print( "The pi value is :", phi)
        return np.array([v, phi])

    def choose_random_control(self):
        controls = [ [1,0],[-1,0] ,[1,1],[1,-1] ,[-1,1],[-1,-1] ]
        control = controls[np.random.choice(len(controls))]
        return np.array(control)

    def is_collision_free_check(self, car_state , sphere_position , sphere_radius):
        r = sphere_radius
        s_x,s_y,s_z = np.array(sphere_position)
        c_x,c_y,c_theta = car_state
        lx,ly,lz = np.array([2,1,1])/2
        local_vertices = np.array([[-lx,-ly,-ly],[lx,-ly,-lz],[-lx,ly,-lz],[lx,ly,-lz],[-lx,-ly,lz],[lx,-ly,lz],[-lx,ly,lz],[lx,ly,lz]])
        rotation_matrix = self.car_robot.quaternion_to_matrix(self.car_robot.rotate_z_quaternion(c_theta))
        transformed_points = np.dot(rotation_matrix, local_vertices.T)
        x_min,y_min,z_min = np.min(transformed_points , axis = 1) + np.array([c_x,c_y,0.5])
        x_max,y_max,z_max = np.max(transformed_points , axis = 1) + np.array([c_x,c_y,0.5])
        # get box closest point to sphere center by clamping
        X = max(x_min , min(s_x , x_max))
        Y = max(y_min , min(s_y , y_max))
        Z = max(z_min , min(s_z , z_max))
        distance  = np.sqrt((X - s_x)*(X - s_x)  + (Y - s_y)*(Y - s_y) + (Z - s_z)*(Z - s_z))

        if distance > r :
            return False
        else:
            return True
        
    def obstacle_collision_check(self , car_state ):
        for obstacle in self.obstacles:
            sphere_position = np.array(obstacle[:3])
            sphere_radius = obstacle[3]
            if self.is_collision_free_check( car_state , sphere_position , sphere_radius):
                return True
        return False    

    def calculate_car_path_without_collision(self, car_controls, start_config , duration ):
        trajectory = self.car_robot.simulate_trajectory( car_controls , start_config , duration)
        for car_state in trajectory:
            if self.obstacle_collision_check(car_state) == True:
                return -1
        return trajectory
    
    def is_goal_region(self, car_state):
        x1, y1, theta1 = car_state
        x2, y2, theta2 = self.goal
        distance_xy = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        distance_theta = min(abs(theta2 - theta1), 2 * math.pi - abs(theta2 - theta1))
        if distance_xy <=  0.1 and distance_theta <= 0.5:
            return True
        else: return False

    def is_valid_node(self, car_state):
        if self.obstacle_collision_check(car_state ) == False:
            return True # no collision
        else:
            return False        

    def rrt(self, max_iterations= 5000):
        for i in range(max_iterations):
            if i % 100 == 0:
                print(i)
            flag_1 = True
            while flag_1:            
                rand_state = self.sample()
                if self.is_valid_node(rand_state) and tuple(rand_state) not in self.tree.list_nodes:
                    flag_1 = False
                    break
            # print(" rand_state",rand_state ,end= ' ')
            nearest_state = self.nearest(list(self.tree.list_nodes.keys()), rand_state)
            # print(" nearest_state",nearest_state ,end= ' ')
            flag_2 = True
            while flag_2:
                control = self.choose_random_control()
                # control = self.choose_guided_control(nearest_state , rand_state)
                duration = np.random.randint(1,7)
                trajectory = self.calculate_car_path_without_collision( control , nearest_state , duration)
                if type(trajectory) != int:
                    flag_2 = False
                    # print("internal trajectory" , trajectory)
                    self.tree.add_child(self.tree.list_nodes[nearest_state] , tuple(control) , trajectory)
                    # print(" control , duration , trajectory[-1]", control , duration , trajectory[-1])
                    break
            if self.is_goal_region(trajectory[-1]):
                # print("goal region reached")
                return self.tree.get_path_to_goal(trajectory[-1])
            
        nearest_state = self.nearest(list(self.tree.list_nodes.keys()), self.goal)
        # print("goal region could not be reached but the path till the neaest point is:")    
        return self.tree.get_path_to_goal(nearest_state)

    def get_visual_trajectory(self,tree_path):
        path , actions = tree_path
        complete_path = []
        for i in range(1,len(path)):
            complete_path.extend(self.tree.branch_configs[tuple([path[i-1] , path[i]])])
       
        self.car_robot.visualize_given_trajectory(complete_path)       
        return complete_path

    def get_tree_visualization(self):
        black = "0x000000"
        all_connections = list(self.tree.branch_configs.values())
        for i,x in enumerate(all_connections):
            x[:,2] = 0.5
            all_connections[i] = x.tolist()
        for i in range(len(all_connections)):
            self.viz_out.add_line(all_connections[i] , black)

if __name__ == "__main__":
    viz_out = threejs_group(js_dir="../js")
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", nargs=3, type=float, required=True)
    parser.add_argument("--goal", nargs=3, type=float, required=True)
    parser.add_argument("--map", type=str, required=True)
    args = parser.parse_args()
    red="0xff0000"
    green="0x00ff00"    
    start = np.array(args.start)
    goal = np.array(args.goal)
    obstacles_file = args.map   
    
    planner = RRTPlanner( viz_out ,start, goal, obstacles_file)
    tree_path = planner.rrt()

    # print(tree_path)

    if tree_path is not None:
        complete_final_path = planner.get_visual_trajectory(tree_path)
        geom = sphere("sphere_0", 1, [start[0] , start[1] , 0.5], [1,0,0,0])
        geom1 = sphere("sphere_1", 1, [goal[0] , goal[1] , 0.5], [1,0,0,0])
        viz_out.add_obstacle(geom, green)
        viz_out.add_obstacle(geom1, red)
        planner.get_tree_visualization()
        viz_out.to_html("rrt_path.html" , "out/")
    else:
        print("No path found.")

    