import networkx as nx

from discrete_optimization.pickup_vrp.gpdp import GPDP


def compute_subtour():
    g_empty = {v: nx.DiGraph() for v in range(vehicle_count)}
    g_merge = nx.DiGraph()
    x_solution = {v: set() for v in range(vehicle_count)}
    model.params.SolutionNumber = s
    for e in x_var:
        value = x_var[e].getAttr("Xn")
        if value >= 0.5:
            vehicle = e[0][0]
            x_solution[vehicle].add(e)
            clients = e[0], e[1]
            if clients[0] not in g_empty[vehicle]:
                g_empty[vehicle].add_node(clients[0])
            if clients[1] not in g_empty[vehicle]:
                g_empty[vehicle].add_node(clients[1])
            if clients[0][1] not in g_merge:
                g_merge.add_node(clients[0][1])
            if clients[1][1] not in g_merge:
                g_merge.add_node(clients[1][1])
            g_empty[vehicle].add_edge(
                clients[0], clients[1], weight=g[e[0]][e[1]]["weight"]
            )
            g_merge.add_edge(
                clients[0][1], clients[1][1], weight=g[e[0]][e[1]]["weight"]
            )
