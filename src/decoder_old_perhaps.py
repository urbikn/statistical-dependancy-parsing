import numpy as np

class CLE():
    """Class implements the graph-based parsing algorithm Chu-Liu-Edmonds.
    """

    def __init__(self, inf=np.inf):
        self.__INF = inf
        
    def __find_one_cycle(self, index_pairs):

        sorted_pairs = np.sort(index_pairs, axis=-1)

        for i, pair in enumerate(sorted_pairs):
            for j, another_pair in enumerate(sorted_pairs[i+1:], start=i+1):
                if np.array_equal(pair, another_pair):
                    return [index_pairs[i], index_pairs[j]]

        return None

    def __decode_graph(self, graph, scores):
        # Graph is actually just a mask matrix made out of either 1 or np.Inf,
        # where 1 is a connection, and np.Inf isn't
        new_scores = np.multiply(scores, graph)
        
        # Find highest-scoring head and return the connection pair
        dependents = np.argmax(new_scores, axis=0)
        index_pairs_with_root = [[h, d] for h, d in enumerate(dependents)]
        index_pairs = index_pairs_with_root[1:]

        # Look for cycles in the graph
        cycle = self.__find_one_cycle(index_pairs)
        if cycle == None:
            return index_pairs
        else:
            pass

            
        return None



    def decode(self, scores):
        # 'scores' holds both scores and what connections are possible
        # so we will use a duplicate for the graph table, but use 1 as a sign
        # for a valid connection and -1 for no connection
        graph = np.where(scores, 1, np.Inf)

        return self.__decode_graph(graph, scores)
        
    