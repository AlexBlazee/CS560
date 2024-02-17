# Generate scenes with spherical objects
# visualize them , load . store them
#space - 100 * 100 * 100

'''
Generate a scene from: i) total number of spheres, ii) minimum radius and iii) maximum radius.
Define your own scene data format.
    def generate_scene(num_spheres, r_min, r_max):
        • Write a scene to a file:
    def scene_to_file(scene, filename):
        • Read a scene from a file:
    def scene_from_file(filename):
        • Visualize a scene using the provided visualization code.
    def visualize_scene(scene) :
'''

from geometry import * 
from threejs_group import *
import numpy as np
import random 
import json

class Scene():
    def __init__(self):
        return
    
    def generate_scene(self , num_spheres , r_min , r_max , sparsity):
        # sparsity = 0.7 # 0.2 means less sparse , 1 means more sparse
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

    def scene_to_file(self , scene , filename) :
        json.dump(scene , open(filename , 'w'))

    def scene_from_file(self , filename):
        scene_dict = json.load(open(filename))
        return scene_dict
    
    def visualize_scene(self , scene , viz_out):
        for i in scene:
            geom = sphere(scene[i]['name'] , scene[i]['radius'] , scene[i]['position'] , scene[i]['orientation'] )
            viz_out.add_obstacle(geom , scene[i]['color'])

if __name__ == "__main__":
    
    scene_params_list = {   1 :{ "num_spheres" : 5 , "r_min" : 3 , "r_max" : 7 , "sparsity" : 0.6 , "file_name" : 'scene_1.txt'},
                            2: { "num_spheres" : 7, "r_min" : 4 , "r_max" : 7 , "sparsity" : 0.7,  "file_name" : 'scene_2.txt'}   ,
                            3: { "num_spheres" : 9, "r_min" : 5 , "r_max" : 8 , "sparsity" : 0.8,  "file_name" : 'scene_3.txt'}   ,
                            4: { "num_spheres" : 11, "r_min" : 4 , "r_max" : 9 , "sparsity" : 0.9,  "file_name" : 'scene_4.txt'}   ,
                            5: { "num_spheres" : 13, "r_min" : 2 , "r_max" : 8 , "sparsity" : 1.0,  "file_name" : 'scene_5.txt'}   ,
                        }

    for i in scene_params_list:
        viz_out = threejs_group(js_dir="../js")
        scene_obj = Scene()
        n_s , r_mi , r_ma , s , f_n = list(scene_params_list[i].values())
        scene_env_dict = scene_obj.generate_scene( n_s , r_mi , r_ma , s)
        scene_obj.scene_to_file(scene_env_dict , f_n)
        scene_obj.visualize_scene(scene_env_dict , viz_out )
        viz_file = "out/viz_" + f_n[:-3] + "html"
        viz_out.to_html(viz_file)

