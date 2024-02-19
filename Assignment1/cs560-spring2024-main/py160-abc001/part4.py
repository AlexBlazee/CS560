import numpy as np  
import quaternion  # Add this import  
import math  
from geometry import *  
from threejs_group import *  
import os
  
class RoboticArm:  
    def __init__(self, viz_out):  
        self.viz_out = viz_out  
  
    def quaternion_rotation(self, axis, angle):  
        q = quaternion.from_rotation_vector(axis * angle)  # Use the quaternion library function  
        return q  
  
    def transform(self, translation, rotation):  
        return (translation, rotation)  
  
    def apply_transform(self, point, transform):  
        t, q = transform  
        return t + quaternion.rotate_vectors(q, point)  # Use the quaternion library function  
  
    def compute_link_pose(self, configuration):  
        theta1, theta2, theta3 = configuration  
  
        base_joint_axis = np.array([0, 0, 1])  
        link1_joint_axis = np.array([0, 1, 0])  
        link2_joint_axis = np.array([0, 1, 0])  
  
        base_joint_rot = self.quaternion_rotation(base_joint_axis, theta1)  
        link1_joint_rot = self.quaternion_rotation(link1_joint_axis, theta2)  
        link2_joint_rot = self.quaternion_rotation(link2_joint_axis, theta3)  
  
        base_pose = self.transform(np.array([0, 0, 0]), base_joint_rot)  
        link1_pose = self.transform(np.array([0, 0, 0.25]), link1_joint_rot)  
        link2_pose = self.transform(np.array([0, 0, 4]), link2_joint_rot)  
  
        link1_world_pose = (base_pose[0] + quaternion.rotate_vectors(base_pose[1], link1_pose[0]), base_pose[1] * link1_pose[1])  # Use the quaternion library function  
        link2_world_pose = (link1_world_pose[0] + quaternion.rotate_vectors(link1_world_pose[1], link2_pose[0]), link1_world_pose[1] * link2_pose[1])  # Use the quaternion library function  
  
        return base_pose, link1_world_pose, link2_world_pose  

  
    def compute_arm_path(self, start_configuration, end_configuration, steps=100):  
        path = []  
        for t in np.linspace(0, 1, steps):  
            config = start_configuration + t * (end_configuration - start_configuration)  
            path.append(self.compute_link_pose(config))  
        return path  
  
    def visualize_arm_path(self, path):  
        base_color = "0xff0000"  
        link1_color = "0x00ff00"  
        link2_color = "0x0000ff"  
  
        base_trajectory = []  
        link1_trajectory = []  
        link2_trajectory = []  
  
        for i, (base_pose, link1_pose, link2_pose) in enumerate(path):  
            base_q = np.array([base_pose[1].x, base_pose[1].y, base_pose[1].z, base_pose[1].w])  # Convert to the expected format  
            link1_q = np.array([link1_pose[1].x, link1_pose[1].y, link1_pose[1].z, link1_pose[1].w])  # Convert to the expected format  
            link2_q = np.array([link2_pose[1].x, link2_pose[1].y, link2_pose[1].z, link2_pose[1].w])  # Convert to the expected format  
  
            base_geom = box(f"base_{i}", 2, 2, 0.5, base_pose[0], base_q)  
            link1_geom = box(f"link1_{i}", 1, 1, 4, link1_pose[0], link1_q)  
            link2_geom = box(f"link2_{i}", 1, 1, 4, link2_pose[0], link2_q)  
  
            base_trajectory.append([i, base_pose[0], base_q, base_color])  
            link1_trajectory.append([i, link1_pose[0], link1_q, link1_color])  
            link2_trajectory.append([i, link2_pose[0], link2_q, link2_color])  
  
        self.viz_out.add_animation(base_geom, base_trajectory)  
        self.viz_out.add_animation(link1_geom, link1_trajectory)  
        self.viz_out.add_animation(link2_geom, link2_trajectory)  
  
        return self.viz_out  
  

if __name__ == "__main__":  
    viz_out = threejs_group(js_dir="../js")  
    robotic_arm = RoboticArm(viz_out)  
  
    start_configuration = np.array([0.0, 0.0, 0.0])  
    end_configuration = np.array([math.pi / 2, math.pi / 4, math.pi / 6])  
  
    path = robotic_arm.compute_arm_path(start_configuration, end_configuration)  
    viz_out = robotic_arm.visualize_arm_path(path)  
  
    # Create output directory if it doesn't exist  
    output_dir = "out"  
    if not os.path.exists(output_dir):  
        os.makedirs(output_dir)  
  
    # Save the HTML file  
    viz_out.to_html("out/robotic_arm_path.html") 

