import networkx as nx
import matplotlib.pyplot as plt
from trees import Tree, kNT, TPT

def get_canonical_label(tree, node, subtree_dict, current_label_dict):
    '''Get the canonical label for the subtree rooted at the given node.'''
    tree.sort_tree(node)
    children = tree.chi(node)
    if not children:
        subtree_form = (tree.node_labels[node]['name'],)
    else:
        subtree_form = (tree.node_labels[node]['name'], tuple(get_canonical_label(tree, child, subtree_dict, current_label_dict) for child in children))

    if subtree_form not in subtree_dict:
        subtree_dict[subtree_form] = current_label_dict['current_label']
        current_label_dict['current_label'] += 1

    return subtree_dict[subtree_form]

def canonize_tree(tree, subtree_dict, current_label_dict):
    '''Canonize the tree and return the canonical form of the root node and the dictionary of subtrees.'''
    canonical_form = get_canonical_label(tree, tree.root, subtree_dict, current_label_dict)
    return canonical_form, subtree_dict

def read_graph_from_file(file_path):
    G = nx.Graph()  # Create an undirected graph
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) < 3:
                continue
            if parts[0] == 'e':
                # Extract the nodes for the edge
                node1 = parts[1]  # Treat node labels as strings
                node2 = parts[2]  # Treat node labels as strings
                # Add the edge to the graph
                G.add_edge(node1, node2)
    return G

if __name__ == "__main__":
    # Example usage
    file_path = 'usr/usr(1)_29-1'  # Replace with your file path
    G = read_graph_from_file(file_path)

    # Plotting the graph
    fig, ax = plt.subplots(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, ax=ax)
    ax.set_title("Undirected Graph from File")
    # plt.show()

    k = 1
    shared_subtree_dict = {}
    current_label_dict = {'current_label': 0}

    knt_canons = []
    tpt_canons = []
    for node in G.nodes():
        knt_instance = kNT(G=G, root=node, k=k)
        tpt_instance = TPT(G, node)

        knt_canon, shared_subtree_dict = canonize_tree(knt_instance, shared_subtree_dict, current_label_dict)
        tpt_canon, shared_subtree_dict = canonize_tree(tpt_instance, shared_subtree_dict, current_label_dict)

        knt_canons.append(knt_canon)
        tpt_canons.append(tpt_canon)

        print('knt:', knt_canon)
        print('tpt:', tpt_canon)

        if (1 == 1): break
