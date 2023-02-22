import numpy as np
from pprint import pprint
import copy

class CLE_n():
    """The class implements the Chu-Liu-Edmonds algorithm, which is a graph-based parsing algorithm used to
    find the maximum spanning tree of a directed graph.

    The argument 'verbose' will print-out all intermediate steps that the algorithm makes.
    """

    def __init__(self, verbose=False, inf=np.inf):
        self.__INF = inf
        self.verbose = verbose
        
    def reverse_graph(self, graph):
        '''Reverses the graph so that graph[head][dep] -> graph[dep][head]'''
        reversed_graph = {}
        for head in graph.keys():
            for dep in graph[head].keys():
                if dep not in reversed_graph.keys():
                    reversed_graph[dep]={}

                reversed_graph[dep][head] = graph[head][dep]

        return reversed_graph


    def find_max_pairs(self, graph):
        '''Find connections that with largest weight for each node'''
        reversed_graph = self.reverse_graph(graph)

        max_graph = {}
        for dep in reversed_graph:
            max_score = -np.Inf
            max_node = None

            for head in reversed_graph[dep]:
                if reversed_graph[dep][head] >= max_score:
                    max_score = reversed_graph[dep][head]
                    max_node = head
            
            if max_node not in max_graph:
                max_graph[max_node] ={}
            
            max_graph[max_node][dep] = max_score

        return max_graph
    

    def matrix_to_graph(self, scores):
        '''Transform adjacency matrix into dictionary, where V= {start node: {end node: cost}}'''
        graph = {}
        for i_start, row in enumerate(scores):
            arcs = {i_end:score for i_end, score in enumerate(row) if score != -np.Inf}

            # Don't include nodes that have no connections
            if len(arcs) > 0:
                graph[i_start] = arcs 
        
        return graph

    def find_cycle(self, graph):
        can_visit = set(graph.keys())
        while can_visit:
            stack = [can_visit.pop()]
            while stack:
                top = stack[-1]
                for node in graph.get(top, ()):
                    if node in stack:
                        cycle_nodes = stack[stack.index(node):] + [node]
                        cycle = {}
                        for i in range(len(cycle_nodes) - 1):
                            head = cycle_nodes[i]
                            dep = cycle_nodes[i + 1]
                            cycle[head] = {dep: graph[head][dep]}
                        
                        return cycle
                    if node in can_visit:
                        stack.append(node)
                        can_visit.remove(node)
                        break
                else:
                    stack.pop()
        return None 

    def contract(self, graph, cycle):
        '''Contracts the cycle by creating new node and updating new graph'''
        new_node_name = max(graph.keys()) + 1
        new_graph = {}
        incoming_original_edge = {}
        outgoing_original_edge = {}

        if self.verbose:
            print('[contract] Starting to create new graph.')

        for head in graph.keys():
            for dep in graph[head].keys():
                # Edge points to a node in the cycle, so we keep the edge with the largest connection
                # == Incoming connection to cycle ==
                if head not in cycle and dep in cycle:
                    # Create new node in new graph
                    if head not in new_graph:
                        new_graph[head] = {}
                    
                    # Get new score of connection as sum from cycle
                    score = graph[head][dep]
                    prev_node, node = dep, next(iter(cycle[dep]))
                    while node != dep:
                        score += graph[prev_node][node]
                        prev_node, node = node, next(iter(cycle[node]))

                    # Creates 'new node' connection, or, has bigger connection update
                    if new_node_name not in new_graph[head] or score > new_graph[head][new_node_name]:
                        new_graph[head][new_node_name] = score
                        incoming_original_edge[head] = dep

                # Edge from cycle to node, so we keep the edge with the largest weight
                # == Outgoing connection to cycle ==
                elif head in cycle and dep not in cycle:
                    # Create new node in new graph
                    if new_node_name not in new_graph:
                        new_graph[new_node_name] = {}
                    
                    score = graph[head][dep]

                    # Creates new connection from 'new node', or, has largest connection from 'new node' to node (dep)
                    if dep not in new_graph[new_node_name] or score > new_graph[new_node_name][dep]:
                        new_graph[new_node_name][dep] = score
                        outgoing_original_edge[dep] = head

                # Edge not connected with cycle
                # == Other edges ==
                elif head not in cycle and dep not in cycle:
                    # Create new node in new graph
                    if head not in new_graph:
                        new_graph[head] = {}

                    new_graph[head][dep] = graph[head][dep]

        if self.verbose:
            print('[contract] New created graph:')
            pprint(new_graph)

        return new_graph, new_node_name, incoming_original_edge, outgoing_original_edge


    def resolve(self, new_graph, new_node_name, cycle, original_graph, incoming_original, outgoing_original):
        for head in list(new_graph.keys()):
            # Find cycle node and resolve it
            if head != new_node_name:
                for dep in list(new_graph[head].keys()):
                    if dep == new_node_name:
                        # Replace with original node
                        new_graph[head].pop(dep)
                        original_dep = incoming_original[head]
                        new_graph[head][original_dep] = original_graph[head][original_dep]

                        # Add nodes from cycle
                        for head_c in cycle:
                            # Except for the one that points to the original node
                            dep_c = next(iter(cycle[head_c]))
                            if dep_c != original_dep:
                                if head_c not in new_graph:
                                    new_graph[head_c] = {}

                                new_graph[head_c][dep_c] = original_graph[head_c][dep_c]
            # Add all out from cycle
            else:
                for dep in list(new_graph[head].keys()):
                    original_head = outgoing_original[dep]
                    if original_head not in new_graph:
                        new_graph[original_head] = {}
                    
                    new_graph[original_head][dep] = original_graph[original_head][dep]

        # Clear out new node from graph
        if new_node_name in new_graph:
            new_graph.pop(new_node_name)
        
        if self.verbose:
            print('[resolve] Resolved graph')
            pprint(new_graph)

        return new_graph


    def __decode_graph(self, graph):
        max_index_pairs = self.find_max_pairs(graph)
        if self.verbose:
            print('[decode] Max index pairs:', max_index_pairs)

        cycle = self.find_cycle(max_index_pairs)
        if self.verbose:
            print('[decode] Cycle found:', cycle)

        if cycle == None:
            if self.verbose:
                print('[decode] No cycle found. Returning max index pairs.')

            return max_index_pairs
        else:
            new_graph, new_node_name, incoming_original, outgoing_original = self.contract(graph, cycle)

            y = self.__decode_graph(copy.deepcopy(new_graph))
            if self.verbose:
                print('[decode] New returned graph')
                pprint(y)

            y_resolved = self.resolve(
                new_graph=y,
                new_node_name=new_node_name,
                cycle=cycle,
                original_graph=graph,
                incoming_original=incoming_original,
                outgoing_original=outgoing_original
            )

            return y_resolved


    def decode(self, scores):
        graph = self.matrix_to_graph(scores)

        max_graph = self.__decode_graph(graph)

        arcs = []
        for head in max_graph:
            for dep in max_graph[head]:
                arcs.append([head, dep])

        return arcs
        

if __name__ == '__main__':
    def test1():
        scores = np.array([
            [-np.Inf,9,10,9],
            [-np.Inf,-np.Inf,20,3],
            [-np.Inf,30,-np.Inf,30],
            [-np.Inf,11,0,-np.Inf]
        ])
        answer = [[2, 3], [0, 2], [2, 1]]

        cle = CLE_n()
        output = cle.decode(scores)

        if sorted(output, key=lambda x: x[1]) != sorted(answer, key=lambda x: x[1]):
            print('Test 1 failed')
            return False
        
        return True

    def test2():
        scores = np.array([
            [-np.Inf, 3, 10, 5],
            [-np.Inf, -np.Inf, 1, 10],
            [-np.Inf, 10, -np.Inf, 8],
            [-np.Inf, 20, 5, -np.Inf],
        ])
        answer =  [[0, 2], [2, 3], [3, 1]]

        cle = CLE_n()
        output = cle.decode(scores)

        if sorted(output, key=lambda x: x[1]) != sorted(answer, key=lambda x: x[1]):
            print('Test 2 failed')
            return False
        
        return True

    def test3():
        scores = np.array([
            [-np.Inf, 47, 64, 67],
            [-np.Inf, -np.Inf, 83, 21],
            [-np.Inf, 87, -np.Inf, 88],
            [-np.Inf, 12, 58, -np.Inf],
        ])
        answer = [[2, 1], [0, 2], [2, 3]]

        cle = CLE_n()
        output = cle.decode(scores)

        if sorted(output, key=lambda x: x[1]) != sorted(answer, key=lambda x: x[1]):
            print(output)
            print('Test 3 failed')
            return False
        
        return True

    def test4():
        np.random.seed(55)
        size = 7
        scores = np.random.randint(0, 100, (size, size)).astype(float)
        scores[:, 0] = -np.Inf
        scores[np.diag_indices_from(scores)] = -np.Inf
        answer = [[2, 1], [6, 2], [0, 3], [3, 4], [3, 5], [3, 6]]


        cle = CLE_n()
        output = cle.decode(scores)

        if sorted(output, key=lambda x: x[1]) != sorted(answer, key=lambda x: x[1]):
            print('Test 4 failed')
            return False
        
        return True


    def test5():
        scores = np.random.randint(0, 30, (10,10)).astype(float)
        scores[:, 0] = -np.Inf
        scores[np.diag_indices_from(scores)] = -np.Inf

        cle = CLE_n()
        output = cle.decode(scores)
        if output == None or len(output) != scores.shape[0] - 1:
            print('Test 5 failed')
            return False
        
        return True

    def test6():
        for i in range(100):
            scores = np.random.randint(0, 30, (100,100)).astype(float)
            scores[:, 0] = -np.Inf
            scores[np.diag_indices_from(scores)] = -np.Inf

            try:
                cle = CLE_n()
                cle.decode(scores)
            except:
                print('Test 5 failed')
                return False
        
        return True

    for test in [test1, test2, test3, test4, test5, test6]:
        result = test()
        if not result:
            exit()
        
    print('All tests run successfully')