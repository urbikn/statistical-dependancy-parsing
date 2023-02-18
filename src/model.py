import numpy as np
import random
import pprint
from tqdm.auto import tqdm
import itertools
import pickle, gzip
import time
from multiprocessing import Pool

import torch
from transformers import AutoTokenizer, AutoModel

from src import decoder
from src import evaluation

class AveragePerceptron:
    def __init__(self, extractor, dim, lambda_val=0.1) -> None:
        self.weight = np.zeros(dim, dtype='float32')
        self.bias = np.random.random()
        self.decoder = decoder.CLE_n()
        self.extractor = extractor
        random.seed(10)

    def train(self, dataset, dev_dataset=None, epoch=5, eval_interval=200, learning_rate=0.1, lambda_val = 0.1, save_folder=None):
        '''
            dataset: [x, true_arcs]
        '''


        print(f'Start training at {epoch} epochs')
        print('Lambda val:', lambda_val)
        best_uas = 0
        for e in range(epoch):
            random.shuffle(dataset)

            progressbar_params = {
                'postfix': {
                    'UAS_train': 0,
                    'UAS_dev': 0,
                    'hit': 0,
                },
                'desc': f'Train {e} epochs',
            }
            hit = 0
            cached_weights = []
                

            with tqdm(enumerate(dataset, start=1), total=len(dataset), **progressbar_params) as progressbar:
                pred_trees = []
                for l, (instance, tree) in progressbar:
                    tree_pred = self.predict(instance)
                    pred_trees.append(tree_pred)


                    # Update weights
                    for arc in tree:
                        indexes = instance[arc[0]][arc[1]]
                        self.weight[indexes] += learning_rate - lambda_val * self.weight[indexes]
                    for arc in tree_pred:
                        indexes = instance[arc[0]][arc[1]]
                        self.weight[indexes] -= learning_rate + lambda_val * self.weight[indexes]



                    # at the end of each batch cache weights and run evaluation
                    if l % eval_interval == 0:
                        cached_weights.append(self.weight)
                        # Run evaluation
                        gold_trees = [tree for instance, tree in dataset[l-eval_interval:l]]
                        uas = evaluation.uas(gold_trees, pred_trees)
                        hit += sum([1 for tree, tree_pred in zip(gold_trees, pred_trees) if sorted(tree, key=lambda x: x[1]) == sorted(tree_pred, key=lambda x: x[1])])
                        progressbar_params['postfix'].update({'UAS_train': uas, 'hit': hit})
                        progressbar.set_postfix(progressbar_params['postfix'])

                        # Restart values
                        pred_trees = []

                    # Development evaluation
                    if dev_dataset is not None and l == len(dataset):
                        progressbar.set_description('== Evaluation ==')
                        gold_dev = [sorted(tree, key=lambda arc: arc[1]) for _, tree in dev_dataset]
                        gold_pred_dev = [sorted(self.predict(x), key=lambda arc: arc[1]) for x, tree in dev_dataset]
                        uas = evaluation.uas(gold_dev, gold_pred_dev)
                        progressbar_params['postfix']['UAS_dev'] = uas
                        progressbar.set_postfix(progressbar_params['postfix'])
                        progressbar.set_description(progressbar_params['desc'])

                        if best_uas < uas:
                            best_uas = uas

                            if save_folder != None:
                                AveragePerceptron.save(self, f"{save_folder}/perceptron-epoch={e}-eval={np.round(uas, 3)}-lambda={np.round(lambda_val, 3)}.p")
                

            # Update weights using cache
            new_weight = np.zeros(self.weight.shape)
            for weight in cached_weights:
                new_weight = np.add(new_weight, weight)

            self.weight = new_weight / len(cached_weights)


        print('Finished training.')

    def predict(self, instance):
        scores = self.forward(instance)
        y_pred = self.decoder.decode(scores)
        return y_pred


    def forward(self, x):
        # +1 right now, because the indexes saved start at 1, but we also have 0
        self.weight[0] = 0
        
        vector = np.zeros((len(x), len(x[0])), dtype="float32")
        for head_index in range(vector.shape[0]):
            for dep_index in range(vector.shape[1]):
                if head_index != dep_index:
                    score = self.weight[x[head_index][dep_index]].sum()
                    vector[head_index][dep_index] = score

        
        # Set scores that shouldn't be computed by the decoder
        vector[:, 0] = -np.inf
        vector[np.diag_indices_from(vector)] = -np.inf

        return vector

    @classmethod
    def save(cls, obj, outfile):
        with gzip.open(outfile,'wb') as stream:
            pickle.dump(obj,stream,-1)

    @classmethod
    def load(cls, inputfile):
        with gzip.open(inputfile,'rb') as stream:
            model = pickle.load(stream)

        return model
