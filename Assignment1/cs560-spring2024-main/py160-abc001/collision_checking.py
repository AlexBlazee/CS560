'''
def check_collision(box, sphere):
(...)
def visualize_collisions(scene):
(...)
'''

from geometry import * 
from threejs_group import *
import random
import numpy as np
import json
from create_scene import Scene

class CollisionCheck():
    def __init__(self , viz_out) -> None:
        self.viz_out = viz_out
        return

    # def add_cube_to_scene(self):
    #     blue="0x0000ff"
    #     geom =  box("box_0", 5,5,5, [random.randint(-50,50),random.randint(-50,50),random.randint(-50,50)], [1,0,0,0])
    #     self.viz_out.add_obstacle(geom , blue)
    #     return geom

    def get_distance_bw_points(self, point_1 , point_2):
        return np.sqrt(np.sum(np.square(point_2 - point_1)))

    def collision_check(self , box , sphere):
        r = sphere.radius
        sphere_pos = np.array(sphere.position)
        s = box.width # since its a cube
        box_pos = np.array(box.position)   

        distance = self.get_distance_bw_points(box_pos , sphere_pos)
        if distance > ( (s*np.sqrt(3))/2 + r):
            return False
        if distance < s/2 or distance < r :
            return True

        # building the equations of the sides of the cube as well as the limits of x,y,z
        x,y,z = [box_pos[0] - s/2 , box_pos[0] + s/2],[box_pos[1] - s/2 , box_pos[1] + s/2],[box_pos[2] - s/2 , box_pos[2] + s/2]

        # assume the line from a cube pos to sphere position of the form <a,b,c,>"sphere" + t * <p-a,q-b,r-c>
        # calculate the t and check if x,y,z fall in the limits
        # check the normals of the vector with the sides of the cube
        t = {'x':[] , 'y': [] , 'z': []}
        vector_direction = sphere_pos - box_pos
        if np.dot(vector_direction , [x[0],0,0]) != 0:
            t['x'].append((x[0] - box_pos[0])/(sphere_pos[0]-box_pos[0]))
            t['x'].append((x[1] - box_pos[0])/(sphere_pos[0]-box_pos[0]))
        if np.dot(vector_direction , [0,y[0],0]) != 0:
            t['y'].append((y[0] - box_pos[1])/(sphere_pos[1]-box_pos[1]))
            t['y'].append((y[1] - box_pos[1])/(sphere_pos[1]-box_pos[1]))
        if np.dot(vector_direction , [0,0,z[0]]) != 0:
            t['z'].append((z[0] - box_pos[2])/(sphere_pos[2]-box_pos[2]))
            t['z'].append((z[1] - box_pos[2])/(sphere_pos[2]-box_pos[2]))

        valid_points = set()
        for key in t:
            if len(t[key]) != 0:
                for val in t[key]:
                    x_val,y_val,z_val = box_pos + val* vector_direction
                    if x_val >= min(x) and x_val <= max(x) and y_val >= min(y) and y_val <= max(y) and z_val >= min(z) and z_val <= max(z):                    
                        if np.dot(np.array([x_val , y_val , z_val]) - box_pos , vector_direction) > 0 :                        
                            valid_points.add(tuple([x_val , y_val , z_val]))
        
        #distance from the point of intersection of cube to center
        distance_picc  = self.get_distance_bw_points(np.array(valid_points.pop()),box_pos)
        if distance > distance_picc + r :
            return False
        else :
            return True

    def visualize_collision(self , scene_dict):

        red="0xff0000"
        green="0x00ff00"    
        blue="0x0000ff"
        num_time_steps = 100
        step_value = 1

        cube_trajectory = []
        spheres_trajector = {}
        for i in scene_dict:
            spheres_trajector[i] = []

        collided_list = []

        for t in np.arange( 0 , num_time_steps , step_value , dtype= float):
            cube_rand_pos = [random.randint(-50 , 50), random.randint(-50 , 50), random.randint(-50 , 50) ]
            cube_state = [t, cube_rand_pos , [1,0,0,0] , blue ]
            cube_trajectory.append(cube_state)
            cube = box("box0", 5,5,5, cube_state[1], cube_state[2])
            # check collision with evry sphere . change color , change and change state
            for i in scene_dict:
                n_s,r_s,p_s,o_s,c_s = list(scene_dict[i].values())
                sp_geom = sphere(n_s , r_s , p_s , o_s )
                if i not in collided_list:
                    if self.collision_check( cube , sp_geom):  
                        collided_list.append(i)
                        sp_geom_state = [t , p_s , o_s , red]  
                    else:   sp_geom_state = [t , p_s , o_s , green]
                else: sp_geom_state = [t , p_s , o_s , red]
                spheres_trajector[i].append(sp_geom_state)
        
        cube = box("box0", 5,5,5, cube_trajectory[0][1], cube_trajectory[0][2])
        self.viz_out.add_animation(cube , cube_trajectory)
        for i in scene_dict:
            n_s,r_s,p_s,o_s,_ = list(scene_dict[i].values())
            sp_geom = sphere(n_s , r_s , spheres_trajector[i][0][1] , spheres_trajector[i][0][2])
            self.viz_out.add_animation(sp_geom , spheres_trajector[i])
    
        return self.viz_out
        # viz_out.add_animation for all the spheres 




if __name__ == "__main__":
    
    file_names = ["scene_1.txt" , "scene_2.txt"  ,"scene_3.txt"  ,"scene_4.txt"  ,"scene_5.txt" ]

    for file in file_names:
        viz_out = threejs_group(js_dir="../js")
        scene_obj = Scene(viz_out)
        scene_dict = scene_obj.scene_from_file(file)
        collision = CollisionCheck(viz_out)
        viz_out = collision.visualize_collision(scene_dict)
        animation_file_name = file[:-4]+ "_animation.html"
        viz_out.to_html(animation_file_name, "out/")

