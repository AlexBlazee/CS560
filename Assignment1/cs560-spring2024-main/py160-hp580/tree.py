class TreeNode:
    def __init__(self, config , parent=None , parent_action = None) :
        self.config = config
        self.parent = parent
        self.parent_action = parent_action
        self.children = {}

    def add_child(self, action , child_config ):
        child = TreeNode(child_config , self , action )
        self.children[tuple(action)] = child  
        return child
      
    def print_config(self):
      print( f"config : {self.config} ")

class Tree:
    
    def __init__(self, config):
        self.root = TreeNode(config)
        self.list_nodes =  { config : self.root }
        self.branch_configs = {}

    def get_child(self, node, action):
        if action in node.children:
            return node.children[action]
        else:
            return None

    def add_child(self, node, action, trajectory):
        ch_node = node.add_child(action ,tuple(trajectory[-1]))
        self.branch_configs[tuple([node.config , ch_node.config])] = trajectory
        self.list_nodes[tuple(trajectory[-1])] = ch_node
        return ch_node

    def get_path_to_goal(self, goal_config):
        path = []
        actions = []
        current_node = self.list_nodes[tuple(goal_config)]
        while current_node.parent is not None:
            path.append(current_node.config)
            actions.append(current_node.parent_action)
            current_node = current_node.parent
        path.append(current_node.config)
        path.reverse()
        actions.reverse()
        return [path , actions]

if __name__ == "__main__":
    # Create a new tree with the root node
    tree = Tree((0, 0, 0))
    # Add a child node to the root
    child1 = tree.add_child(tree.root, (1, 2), [(1, 0, 0), (1, 0, 1), (1, 1, 1)])
    # Add another child node to the root
    child3 = tree.add_child(child1, (0, 0), [(1, 1, 1), (1, 2, 1), (2, 2, 1)])
    child2 = tree.add_child(tree.root, (3, 4), [(2, 0, 0), (2, 0, 2), (2, 2, 2)])

    # Get the child node based on the action
    child = tree.get_child(tree.root, (1, 2))
    if child:
        print(f"Child node: ({child.config})")
    else:
        print("Child node not found")

    # Get the path to a goal node
    goal_config = (2, 2, 1)
    path, actions = tree.get_path_to_goal(goal_config)
    if path:
        print(f"Path from root to goal: {path}")
        print(f"Actions taken: {actions}")
    else:
        print("Goal node not found in the tree")