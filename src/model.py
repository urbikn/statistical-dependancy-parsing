import numpy as np
import random
from tqdm import tqdm
import pickle, gzip

from src import decoder
from src import evaluation

class AveragePerceptron:
    def __init__(self, dim) -> None:
        weight_init = 20

        self.weight = np.random.randint(0, weight_init, dim).astype('float')
        self.decoder = decoder.CLE()

    def train(self, dataset, epoch=5, batch_n=128):
        '''
            dataset: [x, true_features, true_arcs]
        '''

        print(f'Start training at {epoch} epochs')

        for e in range(epoch):
            gold = []
            predictions = []

            batches = []
            for s in range(0, len(dataset), batch_n):
                _batch = []
                for i in range(s, s+batch_n):
                    if i < len(dataset):
                        _batch.append(dataset[i])
                batches.append(_batch)
            random.shuffle(batches)

            for batch in tqdm(batches):
                # Variables used for averaging the weights
                cached_weight = np.zeros(self.weight.shape)
                for x, true_features, y in batch:
                    cached_weight = np.add(cached_weight, self.weight)

                    scores = self.forward(x)
                    scores[:, 0] = -np.inf
                    scores[np.diag_indices_from(scores)] = -np.inf

                    y_pred = self.decoder.decode(scores)

                    # Sort just so it's easier to compare
                    y.sort(key=lambda x: x[1])
                    y_pred.sort(key=lambda x: x[1])

                    gold.append(y)
                    predictions.append(y_pred)

                    if y != y_pred:
                        delta_weight = np.zeros(self.weight.shape, dtype=float)

                        for i, y_arc in enumerate(y):
                            delta_weight = np.add(delta_weight, true_features[i])
                        
                        for i, y_pred_arc in enumerate(y_pred):
                            head, dep = y_pred_arc
                            delta_weight = np.subtract(delta_weight, x[head][dep])
                        

                        self.weight = np.add(self.weight, delta_weight)

                # Update weights with cached weights
                self.weight = np.subtract(self.weight, (1/len(batch)) * cached_weight)
            print('UAS:', evaluation.uas(gold, predictions))

        print('Finished training.')

    def forward(self, x):
        h = np.dot(x, self.weight) + 0.7
        return np.squeeze(h)

    @classmethod
    def save(cls, obj, outfile):
        with gzip.open(outfile,'wb') as stream:
            pickle.dump(obj,stream,-1)

    @classmethod
    def load(cls, inputfile):
        with gzip.open(inputfile,'rb') as stream:
            model = pickle.load(stream)

        return imodel
