import numpy as np
from geometry import box
from threejs_group import threejs_group as viz_group

class RoboticArm:
    def __init__(self, visualization):
        self.visualization = visualization

    def apply_transformation(self, position, quaternion, point):
        rotation_matrix = np.array([
            [1 - 2 * quaternion[2]**2 - 2 * quaternion[3]**2, 2 * quaternion[1] * quaternion[2] - 2 * quaternion[0] * quaternion[3], 2 * quaternion[1] * quaternion[3] + 2 * quaternion[0] * quaternion[2]],
            [2 * quaternion[1] * quaternion[2] + 2 * quaternion[0] * quaternion[3], 1 - 2 * quaternion[1]**2 - 2 * quaternion[3]**2, 2 * quaternion[2] * quaternion[3] - 2 * quaternion[0] * quaternion[1]],
            [2 * quaternion[1] * quaternion[3] - 2 * quaternion[0] * quaternion[2], 2 * quaternion[2] * quaternion[3] + 2 * quaternion[0] * quaternion[1], 1 - 2 * quaternion[1]**2 - 2 * quaternion[2]**2]
        ])
        transformed_point = np.dot(rotation_matrix, point) + position
        return transformed_point.tolist()

    def calculate_forward_kinematics(self, configuration):
        theta1, theta2, theta3 = configuration

        base_dimensions = [2, 2, 0.5]
        link1_dimensions = [1, 1, 4]
        link2_dimensions = [1, 1, 4]

        base_position = [0, 0, base_dimensions[2] / 2]
        base_orientation = [np.cos(theta1/2), 0, 0, np.sin(theta1/2)]

        j1_position = [0, 0, base_dimensions[2]]
        link1_orientation = [
            np.cos((theta1 + theta2)/2),
            np.sin((theta1 + theta2)/2),
            0,
            0
        ]
        link1_position = self.apply_transformation(
            j1_position, link1_orientation, [0, 0, link1_dimensions[2] / 2])

        j2_position = self.apply_transformation(
            j1_position, link1_orientation, [0, 0, link1_dimensions[2]])
        link2_orientation = [
            np.cos((theta1 + theta2 + theta3)/2),
            np.sin((theta1 + theta2 + theta3)/2),
            0,
            0
        ]
        link2_position = self.apply_transformation(
            j2_position, link2_orientation, [0, 0, link2_dimensions[2] / 2])

        transformations = [
            (base_position, base_orientation),
            (link1_position, link1_orientation),
            (link2_position, link2_orientation)
        ]

        return transformations

    def compute_arm_path(self, start_configuration, end_configuration, steps=100):
        path = []
        for t in np.linspace(0, 1, steps):
            interpolated_configuration = tuple(
                np.array(start_configuration) + t * (np.array(end_configuration) - np.array(start_configuration)))
            path.append(interpolated_configuration)
        return path

    def visualize_arm_path(self, start_configuration, end_configuration):
        viz_output = viz_group(js_dir="../js")

        path = self.compute_arm_path(start_configuration, end_configuration)

        boxes = {
            'link0': box("base", 2, 2, 0.5, [0, 0, 0], [np.cos(0/2), 0, 0, np.sin(0/2)]),
            'link1': box("link1", 1, 1, 4, [0, 0, 0], [
                np.cos((start_configuration[0] + start_configuration[1])/2),
                np.sin((start_configuration[0] + start_configuration[1])/2),
                0,
                0
            ]),
            'link2': box("link2", 1, 1, 4, [0, 0, 0], [
                np.cos((start_configuration[0] + start_configuration[1] + start_configuration[2])/2),
                np.sin((start_configuration[0] + start_configuration[1] + start_configuration[2])/2),
                0,
                0
            ])
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

        viz_output.to_html("robotic_arm_path.html", "out/")


if __name__ == "__main__":
    modified_viz_output = viz_group(js_dir="../js")
    modified_robotic_arm = RoboticArm(modified_viz_output)

    initial_configuration = np.random.uniform(-np.pi, np.pi, 3)
    final_configuration = np.random.uniform(-np.pi, np.pi, 3)

    modified_robotic_arm.visualize_arm_path(initial_configuration, final_configuration)
