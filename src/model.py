import numpy as np
import random
from tqdm.auto import tqdm
import pickle, gzip
import time

from src import decoder
from src import evaluation

class AveragePerceptron:
    def __init__(self, extractor, dim, weight_init=20) -> None:
        self.weight = np.random.randint(-weight_init, weight_init, dim).astype('float')
        self.bias = np.random.random()
        self.decoder = decoder.CLE()
        self.extractor = extractor

    def train(self, dataset, dev_dataset=None, epoch=5, batch_n=64, eval_interval=200, learning_rate=0.001):
        '''
            dataset: [x, true_features, true_arcs]
        '''

        print(f'Start training at {epoch} epochs')


        for e in range(epoch):
            random.shuffle(dataset)

            gold = []
            predictions = []
            progressbar_params = {
                'postfix': {
                    'UAS_train': 0,
                    'UAS_dev': 0,
                    'hit': 0,
                },
                'desc': f'Train {e} epochs',
            }
            hit = 0

            weight_cache = []
            # for x, true_features, y in batch:
            with tqdm(enumerate(dataset, start=1), total=len(dataset), **progressbar_params) as progressbar:
                for l, (instance, tree) in progressbar:
                    # Train evaluation
                    if l % batch_n == 0:
                        uas = evaluation.uas(gold, predictions)
                        progressbar_params['postfix']['UAS_train'] = uas
                        progressbar_params['postfix']['hit'] = hit
                        progressbar.set_postfix(progressbar_params['postfix'])
                        gold = []
                        predictions = []
                    
                    # Development evaluation
                    if dev_dataset is not None and l % eval_interval == 0 or l == len(dataset):
                        progressbar.set_description('== Evaluation ==')
                        gold_dev = [tree for x, tree in dev_dataset]
                        gold_pred_dev = [self.predict(x) for x, tree in dev_dataset]
                        uas = evaluation.uas(gold_dev, gold_pred_dev)
                        progressbar_params['postfix']['UAS_dev'] = uas
                        progressbar.set_postfix(progressbar_params['postfix'])
                        progressbar.set_description(progressbar_params['desc'])


                    x = self.extractor.feature_to_tensor(instance)
                    scores = self.forward(x)
                    tree_pred = self.decoder.decode(scores)

                    # Sort just so it's easier to compare
                    tree.sort(key=lambda x: x[1])
                    tree_pred.sort(key=lambda x: x[1])

                    gold.append(tree)
                    predictions.append(tree_pred)

                    if tree != tree_pred:
                        tree = np.array(tree)
                        tree_pred = np.array(tree_pred)

                        pred_features = x[tree_pred[:,0], tree_pred[:,1]]
                        features = x[tree[:,0], tree[:,1]]
                        self.weight = np.add(self.weight, learning_rate * (np.sum(features, axis=0) - np.sum(pred_features, axis=0)))

                        weight_cache.append(self.weight)
                    else:
                        hit += 1

            # Average of all learned weights
            self.weight = np.mean(np.array(weight_cache), axis=0)

        print('Finished training.')

    def predict(self, instance):
        x = self.extractor.feature_to_tensor(instance)
        scores = self.forward(x)
        y_pred = self.decoder.decode(scores)
        return y_pred


    def forward(self, x):
        x_h = np.matmul(x, self.weight)
        # Set scores that shouldn't be computed by the decoder
        x_h[:, 0] = -np.inf
        x_h[np.diag_indices_from(x_h)] = -np.inf

        return x_h

    @classmethod
    def save(cls, obj, outfile):
        with gzip.open(outfile,'wb') as stream:
            pickle.dump(obj,stream,-1)

    @classmethod
    def load(cls, inputfile):
        with gzip.open(inputfile,'rb') as stream:
            model = pickle.load(stream)

        return model
