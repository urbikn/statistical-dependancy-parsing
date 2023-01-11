import decoder
import numpy as np
import random
from tqdm import tqdm


class Perceptron:
    def __init__(self, dim) -> None:
        weight_init = 20

        self.weight = np.random.randint(0, weight_init, dim)
        self.decoder = decoder.CLE()

    def train(self, dataset, epoch=5, alpha=0.01):
        '''
            dataset: [x, true_features, true_arcs]
        '''

        print(f'Start training at {epoch} epochs')
        for e in range(epoch):
            random.shuffle(dataset)
            correct = 0

            for x, true_features, y in tqdm(dataset):
                scores = self.forward(x)
                scores[:, 0] = -np.inf
                scores[np.diag_indices_from(scores)] = -np.inf

                y_pred = self.decoder.decode(scores)

                # Sort just so it's easier to compare
                y.sort(key=lambda x: x[1])
                y_pred.sort(key=lambda x: x[1])

                if y == y_pred:
                    correct += 1
                else:
                    delta_weight = np.sum(alpha * true_features, axis=0).astype('float').round(5)
                    #print('V:', delta_weight)

                    # Now subtract wrong predicted arcs
                    for y_pred_arc in y_pred:
                        if y_pred_arc not in y:
                            head, dep = y_pred_arc
                            y_pred_feature = x[head][dep]
                            feature_weight = np.array(alpha * y_pred_feature).astype('float').round(5)

                            delta_weight = np.subtract(delta_weight, feature_weight)

                    #print(delta_weight)
                    self.weight = np.add(self.weight, delta_weight)
                    
            total = len(dataset)
            print('Accuracy:', correct / total)
            print(correct)
        
        print('Finished training.')

    def forward(self, x):
        h = np.dot(x, self.weight)
        return np.squeeze(h)
