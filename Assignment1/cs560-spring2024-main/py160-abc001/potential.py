import numpy as np
import argparse
from geometry import *
from threejs_group import threejs_group


def quaternion_from_y_rotation(angle):  
    return [np.cos(angle/2), 0, np.sin(angle/2), 0]  
  
  
def quaternion_from_z_rotation(angle):  
    return [np.cos(angle/2), 0, 0, np.sin(angle/2)]  
  
  
def multiply_quaternions(quaternion1, quaternion2):  
    w1, x1, y1, z1 = quaternion1  
    w2, x2, y2, z2 = quaternion2  
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2  
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2  
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2  
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2  
    return [w, x, y, z]  


  
def quaternion_to_rotation_matrix(quaternion):  
    w, x, y, z = quaternion  
    return np.array([  
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],  
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],  
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]])  
  
def apply_transformation(position, quaternion, point):  
    rotation_matrix = quaternion_to_rotation_matrix(quaternion)  
    transformed_point = np.dot(rotation_matrix, point) + position  
    return transformed_point.tolist()  
  
def compute_forward_kinematics(joint_angles):  
    theta1, theta2, theta3 = joint_angles  
  
    base_dimensions = [2, 2, 0.5]   
    link1_dimensions = [1, 1, 4]     
    link2_dimensions = [1, 1, 4]      
  
    base_position = [0, 0, base_dimensions[2] / 2]    
    base_orientation = quaternion_from_z_rotation(theta1)  
  
    joint1_position = [0, 0, base_dimensions[2]]  
    link1_orientation = multiply_quaternions(base_orientation, quaternion_from_y_rotation(theta2))  
    link1_position = apply_transformation(joint1_position, link1_orientation, [0, 0, link1_dimensions[2] / 2])  
  
    joint2_position = apply_transformation(joint1_position, link1_orientation, [0, 0, link1_dimensions[2]])  
    link2_orientation = multiply_quaternions(link1_orientation, quaternion_from_y_rotation(theta3))  
    link2_position = apply_transformation(joint2_position, link2_orientation, [0, 0, link2_dimensions[2] / 2])  
  
    transformations = [  
        (base_position, base_orientation),  # Transformation for the base  
        (link1_position, link1_orientation),  # Transformation for Link 1  
        (link2_position, link2_orientation),  # Transformation for Link 2  
    ]  
  
    return transformations  


  
def find_nearest_obstacle_distance(q, obstacles):  
    min_distance = float('inf')  
    for obstacle in obstacles:  
        obstacle_x, obstacle_y, obstacle_z = obstacle  
        distance = np.linalg.norm(np.array(q)[:3] - np.array([obstacle_x, obstacle_y, obstacle_z]))  
        min_distance = min(min_distance, distance)  
    return min_distance  
  
def compute_attraction_gradient(q, target_position, alpha):  
    q_x, q_y, q_t = q  
    grad_x = alpha * (q_x - target_position[0])  
    grad_y = alpha * (q_y - target_position[1])  
    grad_theta = alpha * (q_t - target_position[2])  
    return np.array([grad_x, grad_y, grad_theta])    
  
def compute_repulsion_gradient(q, obstacles, rho_0, eta):  
    nearest_obstacle_distance = find_nearest_obstacle_distance(q, obstacles)  
    if nearest_obstacle_distance <= rho_0:  
        grad = np.zeros_like(q)  
        for i in range(len(q)):  
            obstacle_grad = np.zeros_like(q)  
            for obstacle in obstacles:  
                obstacle_3d = np.array([obstacle[0], obstacle[1], obstacle[2]])  
                diff = q[i] - obstacle_3d[i]  
                obstacle_grad[i] = eta * (1 - nearest_obstacle_distance / rho_0) * (diff / np.linalg.norm(np.array(q) - obstacle_3d))**2  
            grad += obstacle_grad  
        return grad  
    else:  
        return np.zeros_like(q)  


import numpy as np  
import argparse  
  
def calculate_potential_gradient(q, target_position, obstacles, alpha, eta, rho_0):  
    return compute_attraction_gradient(q, target_position, alpha) + compute_repulsion_gradient(q, obstacles, rho_0, eta)  
  
def find_robot_arm_path(start_configuration, target_position, obstacles, alpha, eta, rho_0, steps=100, learning_rate=0.01, tolerance=1e-5):  
    q = np.array(start_configuration)  
    path = [q]  
    for _ in range(steps):  
        gradient = calculate_potential_gradient(q, target_position, obstacles, alpha, eta, rho_0)  
        q_new = q - learning_rate * gradient  
        path.append(q_new.tolist())  
        if np.linalg.norm(q_new - q) < tolerance:  
            break  
        q = q_new  
    return path  
  
def visualize_robot_arm_path(start_configuration, target_position, obstacles, alpha, eta, rho_0):  

    viz_group = threejs_group(js_dir="../js")  
  
    base_color = "0xFF0000"    
    link1_color = "0x000000"   
    link2_color = "0xFFFF00"   
      
    path = find_robot_arm_path(start_configuration, target_position, obstacles, alpha, eta, rho_0)  
  
    boxes = {  
        'link0': box("base", 2, 2, 0.5, [0, 0, 0], quaternion_from_z_rotation(0)),  
        'link1': box("link1", 1, 1, 4, [0, 0, 0], quaternion_from_y_rotation(0)),  
        'link2': box("link2", 1, 1, 4, [0, 0, 0], quaternion_from_y_rotation(0))  
    }  
    link_colors = {  
        'link0': base_color,  
        'link1': link1_color,  
        'link2': link2_color  
    }  
  
    keyframes = {name: [] for name in boxes}   
  
    for t, configuration in enumerate(path):  
        transformations = compute_forward_kinematics(configuration)  
        for i, transformation in enumerate(transformations):  
            position, quaternion = transformation  
            keyframes[f'link{i}'].append({  
                'time': t,   
                'position': position,    
                'quaternion': quaternion    
            })  
  
    for name, keyframe_data in keyframes.items():  
        animation_data = [(kf['time'], kf['position'], kf['quaternion'], link_colors[name]) for kf in keyframe_data]  
        viz_group.add_animation(boxes[name], animation_data)  
  
    red="0xff0000"  
    geom = sphere("sphere_0", 1, [0, 0, 4], [1, 0, 0, 0])   
    viz_group.add_obstacle(geom, red)   
      
    viz_group.to_html("potential.html", "../out/")  
  
def parse_command_line_arguments():  
    parser = argparse.ArgumentParser(description='Robotic arm potential field navigation.')  
    parser.add_argument('--start', nargs=3, type=float, help='Start configuration of the arm: theta1 theta2 theta3')  
    parser.add_argument('--goal', nargs=3, type=float, help='Goal configuration of the arm: theta1 theta2 theta3')  
    return parser.parse_args()  


def main():
    args = parse_command_line_arguments()
    
    start_configuration = np.array(args.start)
    goal_configuration = np.array(args.goal)

    obstacles = [(0, 0, 4)]  
    alpha = 1.0 
    eta = 1.0 
    rho_0 = 1.0  
    
    visualize_robot_arm_path(start_configuration, goal_configuration, obstacles, alpha, eta, rho_0)

if __name__ == '__main__':
    main()
