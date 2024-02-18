import numpy as np  
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D  
import quaternion
  
class Link:  
    def __init__(self, length, width, height):  
        self.length = length  
        self.width = width  
        self.height = height  
  
class RobotArm:  
    def __init__(self):  
        self.links = [  
            Link(0.5, 2, 2),  
            Link(4, 1, 1),  
            Link(4, 1, 1)  
        ]  
  
    def forward_kinematics(self, angles):  
        theta1, theta2, theta3 = angles  
  
        # Define transformation matrices  
        T_base = np.eye(4)  
        T_base[:3, :3] = quaternion.as_rotation_matrix(quaternion.from_euler_angles(0, 0, theta1))  
          
        T_link1 = np.eye(4)  
        T_link1[:3, :3] = quaternion.as_rotation_matrix(quaternion.from_euler_angles(0, theta2, 0))  
        T_link1[:3, 3] = [self.links[0].length, 0, 0]  
  
        T_link2 = np.eye(4)  
        T_link2[:3, :3] = quaternion.as_rotation_matrix(quaternion.from_euler_angles(0, theta3, 0))  
        T_link2[:3, 3] = [self.links[1].length, 0, 0]  
  
        T_link3 = np.eye(4)  
        T_link3[:3, 3] = [self.links[2].length, 0, 0]  
  
        # Calculate end-effector positions  
        pos_base = T_base[:3, 3]  
        pos_link1 = (T_base @ T_link1)[:3, 3]  
        pos_link2 = (T_base @ T_link1 @ T_link2)[:3, 3]  
        pos_link3 = (T_base @ T_link1 @ T_link2 @ T_link3)[:3, 3]  
  
        return [pos_base, pos_link1, pos_link2, pos_link3]  
  
def compute_arm_path(start_configuration, end_configuration, num_steps=100):  
    start_angles = np.radians(start_configuration)  
    end_angles = np.radians(end_configuration)  
    step_angles = (end_angles - start_angles) / num_steps  
  
    robot_arm = RobotArm()  
    path = [robot_arm.forward_kinematics(start_angles + step_angles * i) for i in range(num_steps + 1)]  
  
    return path  
  
def visualize_arm_path(path):  
    fig = plt.figure()  
    ax = fig.add_subplot(111, projection='3d')  
  
    for positions in path:  
        xs, ys, zs = zip(*positions)  
        ax.plot(xs, ys, zs, 'r-')  
  
    ax.set_xlabel('X')  
    ax.set_ylabel('Y')  
    ax.set_zlabel('Z')  
    plt.show()  
  
start_configuration = [0, 0, 0]  
end_configuration = [90, 45, 30]  
  
# Compute the arm path  
path = compute_arm_path(start_configuration, end_configuration)  
  
# Visualize the arm path  
visualize_arm_path(path)  


