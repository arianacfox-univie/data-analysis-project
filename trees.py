import matplotlib.pyplot as plt # type: ignore
import networkx as nx # type: ignore

class Tree:
    def __init__(self, root):
        self.root = str(root)
        self.vertices = {self.root: []}
        self.edges = []
        self.nodes = {self.root}
        self.node_labels = {self.root: {'name': self.root.split('_')[0]}}
        self.unfolding_tree = nx.DiGraph()

    # def get_neighbors(self, v):
    #     '''Return neighbors of a given node.'''
    #     outgoing_neighbors = list(self.G.neighbors(v))
    #     incoming_neighbors = list(self.G.predecessors(v))
    #     return list(set(outgoing_neighbors + incoming_neighbors))  # Combine and remove duplicates

    def print_tree(self):
        '''Print the tree as text.'''
        print("Tree Structure:")
        for parent, children in self.vertices.items():
            print(f"Parent: {self.node_labels[parent]['name']}")
            if children:
                child_names = [self.node_labels[child]['name'] for child in children]
                print(f"Children: {child_names}")
            else:
                print("No Children")
            print("-" * 20)

    def chi(self, node):
        '''Return all children of a given node.'''
        return self.vertices[node]
    

class kNT(Tree):
    def __init__(self, G, root, k, h=None):
        super().__init__(root)
        self.G = G
        self.k = k
        # The default maximum height is the number of nodes + 1
        if h is None:
            h = len(G.nodes) + 2
        self.max_depth = h
        self.add_vertex(self.root)
        self.build_tree(self.G, self.root, k, h)

    def add_vertex(self, vertex):
        '''Helper function to add vertex in tree creation.'''
        self.vertices[vertex] = []
        self.nodes.add(vertex)
        self.node_labels[vertex] = {'name': vertex.split('_')[0]}
        self.unfolding_tree.add_node(vertex, label=self.node_labels[vertex]['name'])

    def add_edge(self, parent, child):
        '''Helper function to add edge in tree creation.'''
        self.vertices[parent].append(child)
        self.edges.append((parent, child))
        self.nodes.add(child)
        self.node_labels[child] = {'name': child.split('_')[0]}
        self.unfolding_tree.add_edge(parent, child)

    def build_tree(self, G, w, k, h):
        # Create a dictionary to hold the depths of the nodes
        # We need this so we can check the depth condition
        D = {}
        D[w] = 0
        # Create a queue because we are imitating a BFS algorithm
        queue = [w]

        for i in range(1, h + 1):
            # Create an empty list that we will go through on the next height
            next_queue = []
            # For each node in our queue we do the following
            for v in queue:
                # Get the node name from the unique identifier
                original_node = v.split('_')[0]
                # Get all the neighbors of that node
                for u in self.G.neighbors(original_node):
                    # Make sure string
                    u = str(u)
                    # Record the depth that the neighbor is
                    depth_u = i
                    # Check if the depth of the neighbor violates the condition
                    # If it does, continue to the next neighbor without adding to the tree
                    if u in D and depth_u > D[u] + k:
                        continue
                    # If the condition is not violated, add the neighbor to the tree
                    D[u] = min(D.get(u, depth_u), depth_u)
                    unique_child = f"{u}_{i}_{original_node}_{v}"
                    self.add_vertex(unique_child)
                    self.add_edge(v, unique_child)
                    # And add the neighbor to the new queue we will check at the next depth
                    next_queue.append(unique_child)
            queue = next_queue

    def plot_tree(self):
        '''Plot the tree on its own plot.'''
        fig, ax = plt.subplots(figsize=(6, 4))
        pos = nx.drawing.nx_agraph.graphviz_layout(self.unfolding_tree, prog='dot')  # Tree layout
        node_labels = {node: self.node_labels[node]['name'] for node in self.unfolding_tree.nodes()}        
        nx.draw(self.unfolding_tree, pos, ax=ax, labels=node_labels, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700)
        ax.set_title(f'{self.k}-Redundant Tree from Root {self.root}')
        plt.show()

    def plot_tree_with_ax(self, ax):
        '''Plot the tree on a given axis (for printing multiple subplots in one plot).'''
        pos = nx.drawing.nx_agraph.graphviz_layout(self.unfolding_tree, prog='dot')  # Tree layout
        node_labels = {node: self.node_labels[node]['name'] for node in self.unfolding_tree.nodes()}        
        nx.draw(self.unfolding_tree, pos, ax=ax, labels=node_labels, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700)
        ax.set_title(f'{self.k}-Redundant Tree from Root {self.root}')

    def sort_tree(self, node):
        '''Recursively sort the children of each node lexicographically by their label.'''
        children = self.vertices.get(node, [])
        for child in children:
            self.sort_tree(child)
        children.sort(key=lambda x: self.node_labels[x]['name'])
        self.vertices[node] = children

        # Remove and re-add edges to maintain sorted order
        edges_to_remove = [(node, child) for child in self.unfolding_tree.successors(node)]
        self.unfolding_tree.remove_edges_from(edges_to_remove)
        self.unfolding_tree.add_edges_from([(node, child) for child in children])


class TPT(Tree):
    def __init__(self, G, root, max_depth=None, iteration_limit=100000):
        super().__init__(root)
        self.G = G
        # In the testing I don't actually use the max_depth like this because DFS is complex and we can logically restrict easily
        if max_depth is None:
            max_depth = len(G.nodes) + 2
        self.max_depth = max_depth
        self.tpt_tree = nx.DiGraph()

        self.iteration_limit = iteration_limit
        self.build_tree(self.root)  # Use modified BFS to add paths

    def add_node_to_tree(self, node_id, node_name, label):
        '''Helper function to add a node to the tree.'''
        self.tpt_tree.add_node(node_id, label=label)
        self.node_labels[node_id] = {'label': label, 'name': node_name}
        self.nodes.add(node_id)

    def add_edge_to_tree(self, parent_id, current_id):
        '''Helper function to add an edge to the tree.'''
        self.tpt_tree.add_edge(parent_id, current_id)
        self.vertices[parent_id].append(current_id)
        if current_id not in self.vertices:
            self.vertices[current_id] = []

    def build_tree(self, start_node):
        '''Main function used to create the tree using mostly BFS logic.
        Instead of nodes just being visited/not visited, we also must check for the special condition
        of the TPT that the node can have already been visited if it is the root.'''
        # Enqueue the start node until its vertices are visited
        queue = [(start_node, [start_node])]
        iteration_count = 0  # To keep track of the number of iterations

        # Continue as long as there is a queue
        while queue:
            current_node, path = queue.pop(0)
            depth = len(path) - 1

            node_id = "-".join(path)
            node_name = node_id.split('-')[-1]

            # If the depth is 0 then we are at the root so we just add it
            if depth == 0:
                self.add_node_to_tree(node_id, node_name, self.root)
            # Otherwise we need to consider the parent when giving the node a unique name for the tree
            else:
                parent_id = "-".join(path[:-1])
                self.add_node_to_tree(node_id, node_name, current_node)
                self.add_edge_to_tree(parent_id, node_id)

            # We should never reach this condition for our testing, but it is here in case we intentionally set a lower value
            if depth >= self.max_depth:
                continue

            neighbors = self.G.neighbors(current_node)
            # neighbors.sort()

            # Go through all the neighbors of the current step
            for neighbor in neighbors:
                neighbor = str(neighbor)

                # This is the only case where we can have a repeat node
                # It has to be the root and the path length has to be greater than 2
                if neighbor == start_node and len(path) > 2:
                    # new_node_id = "-".join(new_path)
                    new_node_id = f"{node_id}-{neighbor}"
                    new_node_name = new_node_id.split('-')[-1]
                    self.add_node_to_tree(new_node_id, new_node_name, current_node)
                    self.add_edge_to_tree(node_id, new_node_id)
                    break  # Stop further exploration from this path

                # If we haven't seen the node before, queue it for the BFS
                if neighbor not in path:
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))

            # iteration_count += 1
            # if iteration_count > self.iteration_limit:
            #     print("iteration limit reached")
            #     break

    def plot_tree(self):
        '''Plot the tree on its own plot.'''
        fig, ax = plt.subplots(figsize=(9, 3))
        pos = nx.drawing.nx_agraph.graphviz_layout(self.tpt_tree, prog='dot')
        node_labels = {node: self.node_labels[node]['name'] for node in self.tpt_tree.nodes()}
        nx.draw(self.tpt_tree, pos, ax=ax, labels=node_labels, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
        ax.set_title(f'TPT from Root: {self.root}')
        plt.show()

    def plot_tree_with_ax(self, ax):
        '''Plot the tree given an axis (intended for use as subplot in plots).'''
        pos = nx.drawing.nx_agraph.graphviz_layout(self.tpt_tree, prog='dot')
        node_labels = {node: self.node_labels[node]['name'] for node in self.tpt_tree.nodes()}
        nx.draw(self.tpt_tree, pos, ax=ax, labels=node_labels, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
        ax.set_title(f'TPT from Root: {self.root}')

    def sort_tree(self, node):
        '''Recursively sort the children of each node lexicographically by their label.'''
        children = self.chi(node)
        for child in children:
            self.sort_tree(child)
        children.sort(key=lambda x: self.node_labels[x]['label'])
        self.tpt_tree.remove_edges_from([(node, child) for child in children])
        self.tpt_tree.add_edges_from([(node, child) for child in children])

