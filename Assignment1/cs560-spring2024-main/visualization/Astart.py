import heapq
import math
import numpy as np

class Cell:
    def __init__(self) -> None:
        self.parent = [0 , 0 , 0]
        self.f = float('inf')
        self.g = float('inf')
        self.h = 0
    

Grid_Rows  = 3 # 100*100*100 grid with each cell 5*5*5
Grid_Cols  = 3 
Grid_Depth = 3

def check_cell_validity( row_id , col_id , dep_id):
    return row_id >= 0 and row_id < Grid_Rows and col_id >= 0 and col_id < Grid_Cols  and dep_id >= 0 and dep_id < Grid_Depth 

def check_cell_collision( grid , row_id , col_id , dep_id):
    # use the collision checker to build a map of the grid
    return grid[row_id][col_id][dep_id] == 1 

def check_is_goal_state( row_id , col_id , dep_id , goal_state):
    return np.all(np.array(goal_state) == np.array([row_id , col_id , dep_id])) 
    #return goal_state[0] == row_id and goal_state[1] == col_id and goal_state[2] == dep_id

def get_euclidian_heuristic_val (row_id , col_id , dep_id , goal_state):
    return math.sqrt(sum((goal_state - np.array([row_id , col_id  , dep_id])) ** 2))

def get_path( cell_props , goal_state):
    path = []
    row_id,col_id,dep_id = goal_state
    while not (cell_props[row_id , col_id , dep_id].parent == [row_id , col_id , dep_id] ):
        path.append((row_id,col_id,dep_id))
        row_id,col_id,dep_id = cell_props[row_id , col_id , dep_id].parent

    path.append((row_id, col_id , dep_id))
    path.reverse()
    return path

def a_star_search( grid , init_state , goal_state):
    
    # Initial states and Goal state are valid
    if not check_cell_validity(init_state[0], init_state[1] , init_state[2]) or not check_cell_validity(goal_state[0], goal_state[1] , goal_state[2]):
        print("Initial State or Goal State is invalid")
        return
 
    # Check if the source and destination are unblocked
    if not check_cell_collision(grid,init_state[0], init_state[1] , init_state[2]) or not check_cell_collision(grid, goal_state[0], goal_state[1] , goal_state[2]):
        print("Initial State or Goal State is in Collision")
        return
 
    # Check if we are already at the destination
    if check_is_goal_state(init_state[0], init_state[1] , init_state[2] , goal_state):
      print("Already at Goal State")
      return

    closed_list = np.zeros((Grid_Rows,Grid_Cols,Grid_Depth)) > 1 # False array
    cell_props = np.array([[[Cell() for _ in range(Grid_Depth)] for _ in range(Grid_Cols)] for _ in range(Grid_Rows)])

    i,j,k = init_state
    cell_props[i,j,k].f = 0
    cell_props[i,j,k].g = 0
    cell_props[i,j,k].h = 0
    cell_props[i,j,k].parent = [i,j,k]
    
    open_list = []
    heapq.heappush(open_list , (0.0 , i, j, k ))

    at_goal_state = False

    while len(open_list) >0:
        p = heapq.heappop(open_list)
        _,i,j,k = p
        closed_list[i,j,k] = True

        directions = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
        for des in directions:
            new_i, new_j , new_k = i + des[0] , j + des[1] , k + des[2]

            if check_cell_validity(new_i,new_j,new_k) and check_cell_collision(grid , new_i , new_j , new_k) and not closed_list[new_i,new_j,new_k]:
                if check_is_goal_state(new_i , new_j , new_k , goal_state ):
                    cell_props[new_i , new_j , new_k].parent = [i,j,k]
                    # goal state reached
                    final_path = get_path(cell_props , goal_state)
                    at_goal_state = True
                    return final_path
                else:
                    g_new = cell_props[i,j,k].g + 1.0
                    h_new = get_euclidian_heuristic_val(new_i , new_j , new_k , goal_state)
                    f_new = g_new + h_new
                    if cell_props[new_i,new_j,new_k].f == float('inf') or cell_props[new_i,new_j,new_k].f > f_new :
                        heapq.heappush(open_list , (f_new , new_i , new_j , new_k))
                        cell_props[new_i,new_j,new_k].f = f_new
                        cell_props[new_i,new_j,new_k].g = g_new
                        cell_props[new_i,new_j,new_k].h = h_new
                        cell_props[new_i,new_j,new_k].parent = [i,j,k]
    
    if not at_goal_state:
        print(" Path could not be found")


def main():
    # Define the grid (1 for unblocked, 0 for blocked)
 
    grid = np.ones((3,3,3))
    # Define the source and destination
    src = [0,0,0]
    dest = [2,0,0]
 
    # Run the A* search algorithm
    f_path = a_star_search(grid, src, dest)
    print(f_path)
if __name__ == "__main__":
    main()