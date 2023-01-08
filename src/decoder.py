import numpy as np
from pprint import pprint
import copy

class CLE():
    """Class implements the graph-based parsing algorithm Chu-Liu-Edmonds.
    """

    def __init__(self, verbose=False, inf=np.inf):
        self.__INF = inf
        self.verbose = verbose
        

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

        # Update scores on old graph
        for node, arcs in graph.items():
            # Skip nodes that will be turned into a new node
            if node in cycle_scores:
                continue

            for n in arcs.keys():
                if n in cycle_scores:
                    arcs[n] += cycle_scores[n]
        if self.verbose:
            print('[resolve] New updated graph:')
            pprint(graph)


        if self.verbose:
            print('[resolve] Starting to create new graph.')

        # index that will represent the newly created node
        cycle_node_index = min(cycle_scores.keys())
        new_graph = {}
        for node in graph.keys():
            new_nodes = graph[node].copy()

            max_node_val = -1
            for arc in list(new_nodes.keys()):
                # Get the largest value as connection to new node
                if arc in cycle_scores:
                    score = new_nodes.pop(arc)

                    if max_node_val < score:
                        max_node_val = score


            if max_node_val >= 0:
                new_nodes[cycle_node_index] = max_node_val

            new_graph[node] = new_nodes


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
        if self.verbose:
            print('[resolve] New created graph:')
            pprint(new_graph)


        return graph, new_graph

    def __resolve_cycle(self, new_graph, cycle):
        for node_c in cycle.keys():
            for head_g, dep_g in new_graph:
                if cycle[node_c] == dep_g:
                    cycle = [[h,d] for h, d in cycle.items() if h != node_c]
                    resolved_graph = new_graph + cycle
                    if self.verbose:
                        print('[resolve cycle] Resolved cycle', cycle)
                        print('[resolve cycle] Resolved graph')
                        pprint(resolved_graph)

                    return resolved_graph
        
        if self.verbose:
            print('[resolve cycle] No resolution')
        return new_graph

    def __resolve_node(self, old_graph, new_graph, old_cycle, index_pairs):
        resolved_index_pairs = copy.deepcopy(index_pairs)

        # Resolve dep nodes
        for i, (head_node, dep_node) in enumerate(index_pairs):
            new_score = new_graph[head_node][dep_node]

            for old_dep_node, old_score in old_graph[head_node].items():
                # Second condition used in scenario that the old_graph's head_node
                # Contained the same value as one that's inside the cycle
                # Or that the dep_node itself is in the cycle, 
                # then we swap with one in the cycle
                # === As you can probably guess, this is a  product of debugging
                if new_score == old_score and \
                   (old_dep_node not in old_cycle or dep_node in old_cycle):
                    resolved_index_pairs[i][1] = old_dep_node
                    break

        # Resolve head nodes
        for i, (head_node, dep_node) in enumerate(index_pairs):
            # First see if the head_node was the one from the cycle (meaning it was
            # reduced to the head_node index)
            if head_node in old_cycle:
                #if dep_node not in new_graph[head_node]
                new_score = new_graph[head_node][dep_node]
                # Get all nodes in cycle and try to find the one that was used
                # for the new graph
                for cycle_node in old_cycle.keys():
                    # .get() in case the score doesn't even exist in the node
                    old_score = old_graph[cycle_node].get(dep_node, -1)
                    if new_score == old_score:
                        resolved_index_pairs[i][0] = cycle_node
                        break

        return resolved_index_pairs


    def __decode_graph(self, graph):
        max_index_pairs = self.__find_max_pairs(graph)
        if self.verbose:
            print('[decode] Max index pairs:', max_index_pairs)

        cycle = self.__find_cycle(max_index_pairs)
        if self.verbose:
            print('[decode] Cycle found:', cycle)

        if cycle == None:
            if self.verbose:
                print('[decode] No cycle found. Returning max index pairs.')

            return [[h,d] for d, h in max_index_pairs.items()]
        else:
            old_graph, new_graph = self.__resolve(graph, cycle)

            y = self.__decode_graph(copy.deepcopy(new_graph))
            if self.verbose:
                print('[decode] New graph')
                pprint(y)

            y_resolved = self.__resolve_node(old_graph, new_graph, cycle, y)
            if self.verbose:
                print('[decode] New (resolved) graph')
                pprint(y_resolved)

            try:
                return self.__resolve_cycle(y_resolved, cycle)
            except Exception as e:
                print(y_resolved)
                raise e


    def decode(self, scores):
        graph = self.__matrix_to_graph(scores)

        return sorted(self.__decode_graph(graph))
        

if __name__ == '__main__':
    def test1():
        scores = np.array([
            [-np.Inf,9,10,9],
            [-np.Inf,-np.Inf,20,3],
            [-np.Inf,30,-np.Inf,30],
            [-np.Inf,11,0,-np.Inf]
        ])
        answer = [[2, 3], [0, 2], [2, 1]]

        cle = CLE()
        output = cle.decode(scores)

        if output != answer:
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

        cle = CLE()
        output = cle.decode(scores)

        if output != answer:
            print('Test 2 failed')
            return False
        
        return True


    def test3():
        scores = np.random.randint(0, 30, (10,10)).astype(float)
        scores[:, 0] = -np.Inf
        scores[np.diag_indices_from(scores)] = -np.Inf

        cle = CLE()
        output = cle.decode(scores)
        if output == None or len(output) != scores.shape[0] - 1:
            print('Test 3 failed')
            return False
        
        return True

    def test4():
        for i in range(100):
            scores = np.random.randint(0, 30, (10,10)).astype(float)
            scores[:, 0] = -np.Inf
            scores[np.diag_indices_from(scores)] = -np.Inf

            try:
                cle = CLE()
                cle.decode(scores)
            except:
                print('Test 3 failed')
                return False
        
        return True

    for test in [test1, test2, test3, test4]:
        if not test():
            exit
        
    print('All tests run successfully')