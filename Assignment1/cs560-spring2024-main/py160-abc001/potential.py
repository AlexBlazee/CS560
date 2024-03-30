import argparse  
import numpy as np  
from geometry import *   
from threejs_group import *  
from arm_2 import *  
import math  
  
class PotentialPlanner():  
    def __init__(self, start, goal, viz_out) -> None:  
        self.start = np.array(start)  
        self.goal = np.array(goal)  
        self.viz_out = viz_out  
        self.mra = ModifiedRoboticArm(viz_out)  
        self.alpha = 0.1  # Attractive potential coefficient  
        self.eta = 0.01   # Repulsive potential coefficient  
        self.rho0 = 1.0   # Distance of influence for repulsive potential  
        self.obstacle = np.array([0, 0, 4])  # Obstacle position  
        self.epsilon = 1e-6  # Convergence threshold for gradient descent  
  
    def attractive_potential(self, q):  
        return 0.5 * self.alpha * np.linalg.norm(q - self.goal)**2  
  
    def repulsive_potential(self, q):  
        rho = np.linalg.norm(q - self.obstacle)  
        if rho <= self.rho0:  
            return 0.5 * self.eta * (1/rho - 1/self.rho0)**2  
        else:  
            return 0  
  
    def potential_function(self, q):  
        return self.attractive_potential(q) + self.repulsive_potential(q)  
  
    def gradient_descent(self):  
        q = self.start.copy()  
        path = [q]  
        while True:  
            gradient = self.gradient_potential(q)  
            q_next = q - self.epsilon * gradient  
            path.append(q_next)  
            if np.linalg.norm(gradient) < self.epsilon:  
                break  
            q = q_next  
        return path  
  
    def gradient_potential(self, q):  
        delta = 1e-6  
        gradient = np.zeros_like(q)  
        for i in range(len(q)):  
            q_plus = q.copy()  
            q_plus[i] += delta  
            q_minus = q.copy()  
            q_minus[i] -= delta  
            grad_i = (self.potential_function(q_plus) - self.potential_function(q_minus)) / (2 * delta)  
            gradient[i] = grad_i  
        return gradient  
  
    def visualize(self, path):  
        self.mra.visualize_arm_path(None, None, np.array(path), [])  
  
if __name__ == "__main__":  
    parser = argparse.ArgumentParser()  
    parser.add_argument("--start", type=float, nargs="+", required=True)  
    parser.add_argument("--goal", type=float, nargs="+", required=True)  
    args = parser.parse_args()  
  
    viz_out = threejs_group(js_dir="../js")  
    potential_planner = PotentialPlanner(args.start, args.goal, viz_out)  
    path = potential_planner.gradient_descent()  
    potential_planner.visualize(path)  
    viz_out.to_html("potential_planner_solution.html", "out/")  
