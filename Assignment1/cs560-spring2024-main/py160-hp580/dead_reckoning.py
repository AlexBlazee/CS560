import argparse  
import numpy as np  
from data_collection import DataCollection  
from threejs_group import threejs_group  
  
def load_data(filename):  
    data = []  
    with open(filename, 'r') as file:  
        for line in file:  
            line = line.strip().replace('(', '').replace(')', '').replace(',', '').split()  
            data.append([float(x) for x in line])  
    return np.array(data)  


  
def dead_reckoning(odometry_data, start):  
    estimated_poses = [start]  
    current_pose = np.array(start)  
  
    for odo in odometry_data:  
        v, phi = odo  
        x, y, theta = current_pose  
        theta_new = theta + v * phi * 0.1  
        x_new = x + v * np.cos(theta) * 0.1  
        y_new = y + v * np.sin(theta) * 0.1  
        current_pose = np.array([x_new, y_new, theta_new])  
        estimated_poses.append(current_pose)  
  
    return np.array(estimated_poses)  
  
import os  
  
def visualize_dead_reckoning(viz_out, ground_truth, dead_reckoning):  
    green = "0x00ff00"  
    red = "0xff0000"  
    for i in range(len(dead_reckoning) - 2):  
        viz_out.add_line([ground_truth[i], ground_truth[i+1]], green)  
        viz_out.add_line([dead_reckoning[i], dead_reckoning[i+1]], red)  
    out_directory = "../out"  
    if not os.path.exists(out_directory):  
        os.makedirs(out_directory)  
    viz_out.to_html("dead_reckoning.html", out_directory)  

 
  
if __name__ == "__main__":  
    parser = argparse.ArgumentParser()  
    parser.add_argument("--map", type=str, required=True)  
    parser.add_argument("--odometry", type=str, required=True)  
    parser.add_argument("--landmarks", type=str, required=True)  
    parser.add_argument("--plan", type=str, required=True)  
    args = parser.parse_args()  
  
    viz_out = threejs_group(js_dir="../js")  
  
    plan_data = load_data(args.plan)  
    start_state = plan_data[0]  
  
    odometry_data = load_data(args.odometry)  
    ground_truth_data = load_data(f"data/gt_{args.odometry.split('_')[-2]}_{args.odometry.split('_')[-1].split('.')[0]}.txt")  
  
    dead_reckoning_poses = dead_reckoning(odometry_data, start_state)  
  
    visualize_dead_reckoning(viz_out, ground_truth_data, dead_reckoning_poses)  

    #python dead_reckoning.py --map maps/map.txt --odometry data/odometry_1.0_L.txt --landmarks data/landmarks_1.0_L.txt --plan data/plan_1.0_L.txt