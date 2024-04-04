import numpy as np
from geometry import * 
from threejs_group import threejs_group

class RoboticArmTransformer:
    @staticmethod
    def rotate_y_quaternion(theta):
        return [np.cos(theta/2), 0, np.sin(theta/2), 0]

    @staticmethod
    def rotate_z_quaternion(theta):
        return [np.cos(theta/2), 0, 0, np.sin(theta/2)]

    @staticmethod
    def multiply_quaternions(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 + y1*w2 + z1*x2 - x1*z2
        z = w1*z2 + z1*w2 + x1*y2 - y1*x2
        return [w, x, y, z]

    @staticmethod
    def quaternion_to_matrix(q):
        w, x, y, z = q
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]])

class ModifiedRoboticArm:
    def __init__(self, viz_out):
        self.viz_out = viz_out
        self.transformer = RoboticArmTransformer()
        self.box_info = {0 : { "name" : "base" , "size" : [2, 2, 0.5] , "color" : "0xFF0000"},
                    1 : { "name" : "link1" ,"size" : [1, 1, 4  ] , "color" : "0x000000"},
                    2 : { "name" : "link2" ,"size" : [1, 1, 4  ] , "color" : "0xFFFF00"}}

    def apply_transformation(self, position, quaternion, point):
        rotation_matrix = self.transformer.quaternion_to_matrix(quaternion)
        transformed_point = np.dot(rotation_matrix, point) + position
        return transformed_point.tolist()

    def calculate_forward_kinematics(self, configuration):
        theta1, theta2, theta3 = configuration

        base_dimensions = [2, 2, 0.5]
        link1_dimensions = [1, 1, 4]
        link2_dimensions = [1, 1, 4]

        base_position = [0, 0, base_dimensions[2] / 2]
        base_orientation = self.transformer.rotate_z_quaternion(theta1)

        j1_position = [0, 0, base_dimensions[2]]
        link1_orientation = self.transformer.multiply_quaternions(base_orientation, self.transformer.rotate_y_quaternion(theta2))
        link1_position = self.apply_transformation(j1_position, link1_orientation, [0, 0, link1_dimensions[2] / 2])

        j2_position = self.apply_transformation(j1_position, link1_orientation, [0, 0, link1_dimensions[2]])
        link2_orientation = self.transformer.multiply_quaternions(link1_orientation, self.transformer.rotate_y_quaternion(theta3))
        link2_position = self.apply_transformation(j2_position, link2_orientation, [0, 0, link2_dimensions[2] / 2])

        transformations = [
            (base_position, base_orientation),
            (link1_position, link1_orientation),
            (link2_position, link2_orientation)
        ]

        return transformations

    def calculate_arm_path(self, start_configuration, end_configuration, steps=100):
        path = []
        for t in np.linspace(0, 1, steps):
            interpolated_configuration = tuple(
                np.array(start_configuration) + t * (np.array(end_configuration) - np.array(start_configuration)))
            path.append(interpolated_configuration)
        return path

    def visualize_arms(self , configurations):
        # TODO: visualization is not working , giving a file error, CORRECT THE CODE    
        for k ,configuration in enumerate(configurations):
            transformations = self.calculate_forward_kinematics(configuration)
            for i, transformation in enumerate(transformations):
                position, quaternion = transformation
                geom = box(self.box_info[i]["name"] + str(k) , self.box_info[i]["size"][0] , self.box_info[i]["size"][1] ,self.box_info[i]["size"][2] , position , quaternion)
                self.viz_out.add_obstacle(geom , self.box_info[i]["color"])
        return self.viz_out
    
    def visualize_arm_path(self, start_configuration, end_configuration , path , config_nodes):
        
        if type(path) == int:
            path = self.calculate_arm_path(start_configuration, end_configuration)
        else:
            path_line = [[],[],[]]
        
        boxes = {
            'link0': box("base", 2, 2, 0.5, [0, 0, 0], self.transformer.rotate_z_quaternion(0)),
            'link1': box("link1", 1, 1, 4, [0, 0, 0], self.transformer.rotate_y_quaternion(0)),
            'link2': box("link2", 1, 1, 4, [0, 0, 0], self.transformer.rotate_y_quaternion(0))
        }
        link_colors = {
            'link0': "0xFF0000",
            'link1': "0x000000",
            'link2': "0xFFFF00"
        }
        link_colors_list = list(link_colors.values())
        keyframes = {name: [] for name in boxes}
        sphere_counter = 0
        for t, configuration in enumerate(path):
            transformations = self.calculate_forward_kinematics(configuration)
            for i, transformation in enumerate(transformations):
                position, quaternion = transformation
                if config_nodes != None:
                    if any(np.array_equal(np.array(configuration), item) for item in config_nodes): 
                        geom = sphere('s'+str(sphere_counter) , 0.1 , position , [1,0,0,0] )
                        path_line[i].append(position)
                        self.viz_out.add_obstacle(geom , link_colors_list[i])
                    sphere_counter += 1
                keyframes[f'link{i}'].append({
                    'time': t,
                    'position': position,
                    'quaternion': quaternion
                })
        if config_nodes != None:
            self.viz_out.add_line(path_line[0] , link_colors_list[0] )
            self.viz_out.add_line(path_line[1] , link_colors_list[1] )
            self.viz_out.add_line(path_line[2] , link_colors_list[2] )


        for name, keyframe_data in keyframes.items():
            animation_data = [(kf['time'], kf['position'], kf['quaternion'], link_colors[name]) for kf in keyframe_data]
            self.viz_out.add_animation(boxes[name], animation_data)

        return self.viz_out
        

    def aabb_single_obstacle_collision_check(self , configuration , sphere_position , sphere_radius):
        # TODO : Yet to test the code
        #reference : https://developer.mozilla.org/en-US/docs/Games/Techniques/3D_collision_detection\\
        
        transformations = self.calculate_forward_kinematics(configuration)
        # print(f" The transformations are :" , transformations)
        r = sphere_radius
        s_x,s_y,s_z = np.array(sphere_position)
        distance = []
        for i, [position,orientation] in enumerate(transformations):
            lx,ly,lz = np.array(self.box_info[i]["size"])/2
            local_vertices = np.array([[-lx,-ly,-ly],[lx,-ly,-lz],[-lx,ly,-lz],[lx,ly,-lz],[-lx,-ly,lz],[lx,-ly,lz],[-lx,ly,lz],[lx,ly,lz]])
            rotation_matrix = self.transformer.quaternion_to_matrix(orientation)
            transformed_points = np.dot(rotation_matrix, local_vertices.T)
            x_min,y_min,z_min = np.min(transformed_points , axis = 1) + position
            x_max,y_max,z_max = np.max(transformed_points , axis = 1) + position
            # get box closest point to sphere center by clamping
            X = max(x_min , min(s_x , x_max))
            Y = max(y_min , min(s_y , y_max))
            Z = max(z_min , min(s_z , z_max))
            distance.append(np.sqrt((X - s_x)*(X - s_x)  + (Y - s_y)*(Y - s_y) + (Z - s_z)*(Z - s_z)))
        
        # print(f" The collsion check distance : {distance} and radius :{ r}")
        if np.sum(np.array(distance) > r) == 3 :
            return False
        else:
            return True

    def aabb_env_obstacle_collision_check(self , configuration , obstacles_list):
        for obstacle in obstacles_list:
            #x,y,z,r = obstacle
            if self.aabb_single_obstacle_collision_check(configuration , obstacle[:3] , obstacle[3]):
                return True
        return False

    def calculate_arm_path_without_collision(self, start_configuration, end_configuration, obstacles_list, steps=100):
        path = []
        for t in np.linspace(0, 1, steps):
            interpolated_configuration =  np.array(start_configuration) + t * (np.array(end_configuration) - np.array(start_configuration))
            if self.aabb_env_obstacle_collision_check(interpolated_configuration , obstacles_list) == True:
                return -1
            path.append(tuple(interpolated_configuration))
        return path    


if __name__ == "__main__":
    modified_viz_output = threejs_group(js_dir="../js")
    modified_robotic_arm = ModifiedRoboticArm(modified_viz_output)

    initial_configuration = np.random.uniform(-np.pi, np.pi, 3)
    final_configuration = np.random.uniform(-np.pi, np.pi, 3)
    modified_robotic_arm.visualize_arm_path(initial_configuration, final_configuration , -1 , None)
    modified_viz_output.to_html("modified_robotic_arm_path1.html" , "out/")

#command to run : 
# python .\py160-abc001\nearest_neighbors.py --robot arm --target 0 0 0 -k 3 --configs ".\py160-abc001\configs.txt"