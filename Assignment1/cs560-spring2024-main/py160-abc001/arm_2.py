import numpy as np
from geometry import box
from threejs_group import threejs_group as viz_group

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
    def __init__(self, visualization):
        self.visualization = visualization
        self.transformer = RoboticArmTransformer()

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

    def visualize_arms(self , configurations , viz_out):

        viz_output = viz_group(js_dir="../js")
        box_info = {0 : { "name" : "base" , "size" : [2, 2, 0.5] , "color" : "0xFF0000"},
                    1 : { "name" : "link1" ,"size" : [1, 1, 4  ] , "color" : "0x000000"},
                    2 : { "name" : "link2" ,"size" : [1, 1, 4  ] , "color" : "0xFFFF00"}}
                
        for configuration in configurations:
            transformations = self.calculate_forward_kinematics(configuration)
            for i, transformation in enumerate(transformations):
                position, quaternion = transformation
                geom = box(box_info[i]["name"] , box_info[i]["size"][0] , box_info[i]["size"][1] , box_info[i]["size"][2] , position , quaternion)
                viz_output.add_obstacle(geom , box_info[i]["color"])
        
        viz_output.to_html("out\\nearest_neigbours_arm_configs.html")

    def visualize_arm_path(self, start_configuration, end_configuration):
        viz_output = viz_group(js_dir="../js")

        path = self.calculate_arm_path(start_configuration, end_configuration)

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

        keyframes = {name: [] for name in boxes}

        for t, configuration in enumerate(path):
            transformations = self.calculate_forward_kinematics(configuration)
            for i, transformation in enumerate(transformations):
                position, quaternion = transformation
                keyframes[f'link{i}'].append({
                    'time': t,
                    'position': position,
                    'quaternion': quaternion
                })

        for name, keyframe_data in keyframes.items():
            animation_data = [(kf['time'], kf['position'], kf['quaternion'], link_colors[name]) for kf in keyframe_data]
            viz_output.add_animation(boxes[name], animation_data)

        viz_output.to_html("modified_robotic_arm_path.html", "out/")

if __name__ == "__main__":
    modified_viz_output = viz_group(js_dir="../js")
    modified_robotic_arm = ModifiedRoboticArm(modified_viz_output)

    initial_configuration = np.random.uniform(-np.pi, np.pi, 3)
    final_configuration = np.random.uniform(-np.pi, np.pi, 3)

    modified_robotic_arm.visualize_arm_path(initial_configuration, final_configuration)