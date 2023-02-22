import cProfile, pstats
import numpy as np
import src.decoder as decoder

def run_100_sentence():
    scores = np.random.randint(0, 30, (100,100)).astype(float)
    scores[:, 0] = -np.Inf
    scores[np.diag_indices_from(scores)] = -np.Inf

    cle = decoder.CLE_n()
    cle.decode(scores)

def run_200_sentence():
    scores = np.random.randint(0, 30, (200,200)).astype(float)
    scores[:, 0] = -np.Inf
    scores[np.diag_indices_from(scores)] = -np.Inf

    cle = decoder.CLE_n()
    cle.decode(scores)

if __name__ == '__main__':
    with cProfile.Profile(timeunit=.001) as pr:
        pr.runcall(run_100_sentence)

        stats = pstats.Stats(pr)
        stats.print_stats(':CLE', 0.1)

        pr.runcall(run_200_sentence)

        stats = pstats.Stats(pr)
        stats.print_stats(':CLE', 0.1)