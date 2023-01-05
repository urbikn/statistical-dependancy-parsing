import numpy as np

class CLE():
    """Class implements the graph-based parsing algorithm Chu-Liu-Edmonds.
    """

    def __init__(self, inf=np.inf):
        self.__INF = inf
        

    def __find_max_pairs(self, graph):
        '''Find connections that with largest weight for each node'''
        reversed_graph = self.__reverse_graph(graph)

        index_pairs = {}
        for node, edges in reversed_graph.items():
            head_node = max(edges, key=edges.get)
            index_pairs[node] = head_node

        return index_pairs


    def __reverse_graph(self, graph):
        '''Return the reversed graph where g[dst][src]=G[src][dst]'''
        reversed_graph = {}
        for head in graph.keys():
            for dep in graph[head].keys():
                if dep not in reversed_graph.keys():
                    reversed_graph[dep]={}

                reversed_graph[dep][head] = graph[head][dep]

        return reversed_graph

    def __matrix_to_graph(self, scores):
        '''Transform adjacency matrix into dictionary, where V= {start node: {end node: cost}}'''
        graph = {}
        for i_start, row in enumerate(scores):
            arcs = {i_end:score for i_end, score in enumerate(row) if score >= 0}

            # Don't include nodes that have no connections
            if len(arcs) > 0:
                graph[i_start] = arcs 
        
        return graph

    def __find_cycle(self, index_pairs):
        for node in index_pairs.keys():
            path = [node]

            while index_pairs.get(path[-1], None): # Looking at the current node
                head = index_pairs[path[-1]]

                if head == node:
                    # Return path of cycle with their head:dep connections
                    return {head:dep for head, dep in zip(path, path[1:] + [node])}
                elif head in path: # A way to break out if we get into an infinite loop
                    break
                
                path.append(head)
        
        return None

    def __resolve(self, graph, cycle):
        rev_cycle = {v:k for k, v in cycle.items()}

        # Sums scores of the cycle from the graph
        sum_cycle_scores = sum([graph[head][dep] for head, dep in cycle.items()])
        cycle_scores = {head:(sum_cycle_scores - graph[dep][head]) for head, dep in cycle.items()}

        # index that will represent the newly created node
        cycle_node_index = min(cycle_scores.keys())
        new_graph = {}
        for node in graph.keys():
            new_nodes = graph[node].copy()

            max_node_val = -1
            for n in list(new_nodes.keys()):
                # Get the largest value as connection to new node
                if n in cycle_scores:
                    val = new_nodes[n] + cycle_scores[n]
                    if val > max_node_val:
                        max_node_val = val

                    new_nodes.pop(n)

                if max_node_val >= 0:
                    new_nodes[cycle_node_index] = max_node_val

            new_graph[node]  = new_nodes


        new_node = {}
        for node in cycle.keys():
            arcs = new_graph.pop(node)

            for n in list(arcs.keys()):
                # Remove cycle nodes from arcs
                if n in cycle:
                    arcs.pop(n)
                elif n not in new_node or new_node[n] < arcs[n]:
                    new_node[n] = arcs[n]


        new_graph[cycle_node_index] = new_node
        return new_graph

    def __resolve_cycle(self, new_graph, cycle):
        for node_c in cycle.keys():
            for node_g in new_graph.keys():
                if cycle[node_c] == new_graph[node_g]:
                    cycle = [[h,d] for h, d in cycle.items() if h != node_c]
                    return [[h,d] for h, d in new_graph.items()] + cycle


    def __decode_graph(self, graph):
        max_index_pairs = self.__find_max_pairs(graph)

        cycle = self.__find_cycle(max_index_pairs)

        if cycle == None:
            print(max_index_pairs)
            return max_index_pairs
        else:
            print('cycle')
            print(cycle)
            new_graph = self.__resolve(graph, cycle)

            y = self.__decode_graph(new_graph)

            return self.__resolve_cycle(y, cycle)


    def decode(self, scores):
        graph = self.__matrix_to_graph(scores)

        return self.__decode_graph(graph)
        

if __name__ == '__main__':
    # [[0, 1], [1, 3], [1, 2]]
    scores = np.array([
        [-np.Inf,9,10,9],
        [-np.Inf,-np.Inf,20,3],
        [-np.Inf,30,-np.Inf,30],
        [-np.Inf,11,0,-np.Inf]
    ])


    scores = np.array([
        [-np.Inf, 10, 5, 15],
        [-np.Inf, -np.Inf, 20, 15],
        [-np.Inf, 25, -np.Inf, 25],
        [-np.Inf, 30, 10, -np.Inf]
    ])

    # [[0, 1], [1, 2], [2, 3]]
    scores = np.array([
        [-np.Inf,10,3,5],
        [-np.Inf,-np.Inf,10,8],
        [-np.Inf,1,-np.Inf,10],
        [-np.Inf,5,20,-np.Inf]
    ])

    cle = CLE()
    print(cle.decode(scores))

    # Test with random
    scores = np.random.randint(0, 30, (10,10)).astype(float)
    scores[:, 0] = -np.Inf
    scores[np.diag_indices_from(scores)] = -np.Inf

    # TODO: Still getting an error here, where when returning from resolve_cycle, I'm getting back a list
    # instead of a dictionary
    print(scores)
    print(cle.decode(scores))
