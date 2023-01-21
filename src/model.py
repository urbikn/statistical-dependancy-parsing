import numpy as np
import random
from tqdm.auto import tqdm
import pickle, gzip
import pprint

from src import decoder
from src import evaluation

class AveragePerceptron:
    def __init__(self, dim, weight_init=20) -> None:
        # self.weight = np.random.randint(-weight_init, weight_init, dim).astype('float')
        self.weight = np.zeros(dim).astype('float')
        self.bias = np.random.random()
        self.decoder = decoder.CLE()

    def train(self, dataset, epoch=5, batch_n=64, learning_rate=0.001):
        '''
            dataset: [x, true_features, true_arcs]
        '''

        print(f'Start training at {epoch} epochs')

        for e in range(epoch):
            gold = []
            predictions = []

            random.shuffle(dataset)

            # for x, true_features, y in batch:
            for l, (x, true_features, tree) in tqdm(enumerate(dataset, start=1), total=len(dataset), position=0, leave=True):
                if l % batch_n == 0:
                    print(self.weight)
                    print('UAS:', evaluation.uas(gold, predictions))

                y_pred = self.forward(x)
                y_pred[:, 0] = -np.inf
                y_pred[np.diag_indices_from(y_pred)] = -np.inf

                tree_pred = self.decoder.decode(y_pred)

                # Sort just so it's easier to compare
                tree.sort(key=lambda x: x[1])
                tree_pred.sort(key=lambda x: x[1])

                gold.append(tree)
                predictions.append(tree_pred)

                if len(tree) != len(tree_pred):
                    continue
                if tree != tree_pred:
                    tree = np.array(tree)
                    tree_pred = np.array(tree_pred)

                    y_pred_scores = y_pred[tree_pred[:,0], tree_pred[:,1]]
                    y_scores = np.full(y_pred_scores.shape, 1)

                    features = x[tree_pred[:,0], tree_pred[:,1]]


                    #loss = (y_scores - y_pred_scores)
                    #delta = loss.dot(features).astype('float')

                    self.weight = np.add(self.weight, learning_rate * np.sum(true_features - features, axis=0))

                    # self.weight = np.add(self.weight, learning_rate * delta)



                    #for i, y_arc in enumerate(y):
                    #    delta_weight = np.add(delta_weight, true_features[i])
                    
                    #for i, y_pred_arc in enumerate(y_pred):
                    #    head, dep = y_pred_arc
                    #    delta_weight = np.subtract(delta_weight, x[head][dep])
                    # delta = np.dot(delta_weight.T, predicted_feat)
                    # self.weight = np.subtract(self.weight, learning_rate * delta)

            print('UAS:', evaluation.uas(gold, predictions))

        print('Finished training.')

    def predict(self, instance):
        scores = self.forward(instance)
        scores[:, 0] = -np.inf
        scores[np.diag_indices_from(scores)] = -np.inf

        y_pred = self.decoder.decode(scores)
        return y_pred


    def forward(self, x):
        x_h = np.dot(x, self.weight) + self.bias
        return np.squeeze(x_h)

    @classmethod
    def save(cls, obj, outfile):
        with gzip.open(outfile,'wb') as stream:
            pickle.dump(obj,stream,-1)

    @classmethod
    def load(cls, inputfile):
        with gzip.open(inputfile,'rb') as stream:
            model = pickle.load(stream)

        return model
