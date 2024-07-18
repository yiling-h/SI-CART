import matplotlib.pyplot as plt
import networkx as nx


def add_edges(graph, root):
    if root.left:
        split_info = "X" + str(root.feature_index) + "<=" + f"{root.threshold:.2f}"
        graph.add_edge(root, root.left,
                       split_info=split_info)
        add_edges(graph, root.left)
    if root.right:
        split_info = "X" + str(root.feature_index) + ">" + f"{root.threshold:.2f}"
        graph.add_edge(root, root.right,
                       split_info=split_info)
        add_edges(graph, root.right)


def draw_tree(root):
    graph = nx.DiGraph()
    add_edges(graph, root)

    pos = hierarchy_pos(graph, root)
    labels = {edge: graph.edges[edge]['split_info'] for edge in graph.edges}

    plt.figure(figsize=(15, 15))
    nx.draw(graph, pos=pos, with_labels=False, arrows=False, node_size=100, node_color='skyblue')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    plt.show()


def hierarchy_pos(G, root=None, width=2, vert_gap=1, vert_loc=0, xcenter=0.5):
    pos = _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
    return pos


def _hierarchy_pos(G, root, width=2, vert_gap=1, vert_loc=0, xcenter=0.5, pos=None, parent=None, parsed=[]):
    if pos is None:
        pos = {root: (xcenter, vert_loc)}
    else:
        pos[root] = (xcenter, vert_loc)

    children = list(G.neighbors(root))
    if not isinstance(G, nx.DiGraph) and parent is not None:
        children.remove(parent)

    if len(children) != 0:
        dx = width / len(children)
        nextx = xcenter - width / 2 - dx / 2
        for child in children:
            nextx += dx
            pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                 pos=pos, parent=root, parsed=parsed)

    return pos
