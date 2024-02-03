from geometry import * 
from threejs_group import *
import random
import numpy as np
import json

def generate_scene(num_spheres , r_min , r_max):
    sparsity = 0.7 # 0.2 means less sparse , 1 means more sparse
    scene_dict= {}
    for i in range(num_spheres):
        obj_name = 'sphere_'+ str(i)
        rand_radius = random.randint(r_min , r_max)
        rand_position = [random.randint(-50 , 50) * sparsity, random.randint(-50 , 50) * sparsity, random.randint(-50 , 50) * sparsity ]
        rand_color = str(hex(random.randrange(0,2**24)))
        # geom = sphere(obj_name , rand_radius , rand_position , [1,0,0,0])
        scene_dict[i] = { 'name': obj_name , 'radius' : rand_radius , 'position' : rand_position , 'orientation' : [1,0,0,0] , 'color' : rand_color}
        # viz_out.add_obstacle(geom , rand_color)   
    return scene_dict

def scene_to_file(scene , filename) :
    json.dump(scene , open(filename , 'w'))

def scene_from_file(filename):
    scene_dict = json.load(open(filename))
    return scene_dict

def visualize_scene(scene):
    for i in scene:
        geom = sphere(scene[i]['name'] , scene[i]['radius'] , scene[i]['position'] , scene[i]['orientation'] )
        viz_out.add_obstacle(geom , scene[i]['color'])

####### 2nd ###########
def add_cube_to_scene():
    red="0xff0000"
    geom =  box("box_0", 2,2,2, [2,2,2], [1,0,0,0])
    viz_out.add_obstacle(geom , red)
    return geom

def collision_check(box , sphere):
    r = sphere.radius
    sphere_pos = np.array(sphere.position)
    s = box.width # since its a cube
    box_pos = np.array(box.position)   

    distance = get_distance_bw_points(box_pos , sphere_pos)
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
    distance_picc  = get_distance_bw_points(np.array(valid_points.pop()),box_pos)
    if distance > distance_picc + r :
        return False
    else :
        return True

def get_distance_bw_points(point_1 , point_2):
    return np.sqrt(np.sum(np.square(point_2 - point_1)))

######### 2nd ##############


if __name__ == "__main__":
    # red="0xff0000"
    green="0x00ff00"
    # purple="0xff00ff"
    # blue="0x0000ff"
    viz_out = threejs_group(js_dir="../js")
    # scene = generate_scene(10 , 1, 3)
    # scene_to_file(scene , "scene_1.txt")
    scene = scene_from_file("scene_1.txt")
    visualize_scene(scene)
    
    ########## 2nd ###############

    # Test 1 code 
        # cube = add_cube_to_scene()
        # sphere_1 = sphere("sphere_11", 1, [4,3,4], [1,0,0,0])
        # viz_out.add_obstacle(sphere_1 , green)
        # print(collision_check(cube , sphere_1))
    ## End of Test code

    cube_trajectory = []
    num_time_steps = 10
    blue="0x0000ff"
    for t in np.arange( 0 , num_time_steps , 0.1 , dtype= float):
        cube_rand_pos = [random.randint(-50 , 50), random.randint(-50 , 50), random.randint(-50 , 50) ]
        state = [t, cube_rand_pos , [1,0,0,0] , blue ]
        cube_trajectory.append(state)

    cube = box("box0", 1,1,1, cube_trajectory[0][1], cube_trajectory[0][2])
    viz_out.add_animation(cube , cube_trajectory)
    viz_out.to_html("box_animation.html", "out/")
    ########### 2nd ############

    viz_out.to_html("out/spherical_obstacles.html")
