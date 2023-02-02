from pprint import pprint
import pickle, gzip
from collections import defaultdict
from multiprocessing import Pool
from tqdm.auto import tqdm, trange
import numpy as np

from src.dataset import ConllSentence
from src.dataset import ConllDataset


class FeatureMapping:
    def __init__(self) -> None:
        self.feature_extractors = {
            'hform': self.hform,
            'hpos': self.hpos,
            'dform': self.dform,
            'dpos': self.dpos,
            'hform, dpos': self.hform_dpos,
            'hpos, dpos': self.hpos_dform,
            'hform, dform': self.hform_dform,
            'hpos, dpos': self.hpos_dpos,
            'hform, hpos, dform, dpos': self.hformpos_dformpos,
            'hform, hpos, dform': self.hformpos_dform,
            'hform, hpos, dpos': self.hformpos_dpos,
            'hform, dform, dpos': self.hform_dformpos,
            'hpos, dform, dpos': self.hpos_dformpos,
            'hpos, dpos, hpos+1, dpos-1': self.hpos_next_dpos_prev,
            'hpos, dpos, hpos-1, dpos+1': self.hpos_prev_dpos_next,
            'hpos, dpos, hpos+1, dpos+1': self.hpos_next_dpos_next,
            'hpos, dpos, hpos-1, dpos-1': self.hpos_prev_dpos_prev,
        }

        self.feature = defaultdict(int)
        self.frozen = False

    # Feature extractor functions
    # ====
    def distance_direction(self, sentence, index_dep, index_head):
        distance = abs(index_dep - index_head)
        if index_head < 0 or index_dep < 0 or len(sentence) <= index_head or len(sentence) <= index_dep:
            return '__NULL__'
        elif index_head < index_dep:
            return f'left+{distance}'
        else: 
            return f'right+{distance}'
    
    def hform(self, sentence, index_dep, index_head) -> str:
        if index_head < 0 or len(sentence) < index_head:
            form = '__NULL__'
        elif index_head == 0:
            form = 'ROOT'
        else:
            form = sentence[index_head]['form']

        return form

    def hpos(self, sentence, index_dep, index_head) -> str:
        if index_head < 0 or len(sentence) < index_head:
            pos = '__NULL__'
        elif index_head == 0:
            pos = 'ROOT'
        else:
            pos = sentence[index_head]['pos']

        return pos

    def dform(self, sentence, index_dep, index_head) -> str:
        if index_dep <= 0 or len(sentence) < index_dep:
            form = '__NULL__'
        else:
            form = sentence[index_dep]['form']

        return form

    def dpos(self, sentence, index_dep, index_head) -> str:
        if index_dep <= 0 or len(sentence) < index_dep:
            pos = '__NULL__'
        else:
            pos = sentence[index_dep]['pos']

        return pos

    def hform_pos(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.hform(*params)}+{self.hpos(*params)}'

    def dform_pos(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.dform(*params)}+{self.dpos(*params)}'

    def hform_dpos(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.hform(*params)}+{self.dpos(*params)}',

    def hpos_dform(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.hpos(*params)}+{self.dform(*params)}',

    def hform_dform(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.hform(*params)}+{self.dform(*params)}',

    def hform_dform(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.hform(*params)}+{self.dform(*params)}',

    def hpos_dpos(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.hpos(*params)}+{self.dpos(*params)}',

    def hformpos_dformpos(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.hform_pos(*params)}+{self.dform_pos(*params)}',

    def hformpos_dform(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.hform_pos(*params)}+{self.dform(*params)}',

    def hformpos_dpos(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.hform_pos(*params)}+{self.dpos(*params)}',

    def hform_dformpos(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.hform(*params)}+{self.dform_pos(*params)}',

    def hpos_dformpos(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.hpos(*params)}+{self.dform_pos(*params)}',

    def hpos_dformpos(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.hpos(*params)}+{self.dform_pos(*params)}',

    def hpos_next_dpos_prev(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        params_neighbors = (sentence, index_dep - 1, index_head + 1)
        return f'{self.hpos(*params)}+{self.hpos(*params_neighbors)}+{self.dpos(*params)}+{self.dpos(*params_neighbors)}',
        
    def hpos_prev_dpos_next(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        params_neighbors = (sentence, index_dep+1, index_head-1)
        return f'{self.hpos(*params)}+{self.hpos(*params_neighbors)}+{self.dpos(*params)}+{self.dpos(*params_neighbors)}',

    def hpos_next_dpos_next(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        params_neighbors = (sentence, index_dep+1, index_head+1)
        return f'{self.hpos(*params)}+{self.hpos(*params_neighbors)}+{self.dpos(*params)}+{self.dpos(*params_neighbors)}',

    def hpos_prev_dpos_prev(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        params_neighbors = (sentence, index_dep-1, index_head-1)
        return f'{self.hpos(*params)}+{self.hpos(*params_neighbors)}+{self.dpos(*params)}+{self.dpos(*params_neighbors)}',

    # ====

    def __len__(self) -> int:
        return len(self.feature)

    def num_features(self) -> int:
        return len(self.feature_extractors)

    def get_feature(self, sentence, index_dep, index_head):
        '''Get index (dictionary key storing index) of features for the head->dependant given a sentence.'''

        feature = [0] * self.num_features()
        for i, (name, func) in enumerate(self.feature_extractors.items()):
            value = str(func(sentence, index_dep, index_head))
            value += self.distance_direction(sentence, index_dep, index_head)
            name_value = f'{name}: {str(value)}'

            if not self.frozen and name_value not in self.feature:
                # Set index for feature
                self.feature[name_value] += len(self.feature)

            feature[i] = self.feature[name_value]

        return feature
        

    def get(self, sentence: ConllSentence):
        '''Get features of all head->dependant's in the sentence.
        
        The features are indexes retrieved from the self.feature dictionary.
        '''
        features = []
        for index_dep in range(1, len(sentence) + 1):
            index_head = sentence[index_dep]['head']

            feature = self.get_feature(sentence, index_dep, index_head)
            features.append(feature)

        return features

    def get_permutations(self, sentence: ConllSentence, default=0) -> np.ndarray:
        '''Used for getting all possible arc features. Don't add new features'''
        # save variable to see if we should unfreeze
        # Freeze feature extractor
        unfreeze = not self.frozen
        self.frozen = True

        features = []
        for index_head in range(0, len(sentence) + 1):
            # Sets the first feature to the default value (to denote all the )
            feature = [np.full(self.num_features(), 0)]
            for index_dep in range(1, len(sentence) + 1):
                if index_head != index_dep:
                    f = self.get_feature(sentence, index_dep, index_head)
                else:
                    f = np.full(self.num_features(), default)
                
                feature.append(f)

            features.append(feature)

        self.frozen = False if unfreeze else self.frozen
        return np.asarray(features)

    def feature_to_tensor(self, feature) -> np.ndarray:
        '''Given a sentence features convert to binary vectors where indexes are 1 and others 0.
        The binary vector is the size of len(self) a.k.a len(self.feature).
        
        Parameters:
            feature: a permutation of a sentence.
            features: A 3D array, where the first dimension is each sentence, and all other from permutations.
        '''
        feature_index = feature
        # +1 right now, because the indexes saved start at 1, but we also have 0
        vector = np.zeros((*feature_index.shape[:-1], len(self) + 1))
        for head_index in range(vector.shape[0]):
            for dep_index in range(vector.shape[1]):
                indexes = feature_index[head_index][dep_index]
                vector[head_index][dep_index][indexes] = 1
                vector[head_index][dep_index][0] = 0
        
        return vector


    def features_to_tensors(self, features) -> np.ndarray:
        '''Given a list of features (so indexes to features) convert to binary vectors where indexes are 1 and others 0.
        The binary vector is the size of len(self) a.k.a len(self.feature).
        
        Parameters:
            features: A 4D array, where the first dimension is each sentence, and all other from permutations.
        '''

        vectors = []
        for i, feature in tqdm(enumerate(features, start=1), total=len(features), desc="Converting features to tensors"):
            feature_index, arcs = feature
            # +1 right now, because the indexes saved start at 1, but we also have 0
            vector = np.zeros((*feature_index.shape[:-1], len(self) + 1))
            for head_index in range(vector.shape[0]):
                for dep_index in range(vector.shape[1]):
                    indexes = feature_index[head_index][dep_index]
                    vector[head_index][dep_index][indexes] = 1
                    vector[head_index][dep_index][0] = 0

            vectors.append((vector, arcs))

        return vectors

        
    @classmethod
    def train_on_dataset(cls, dataset):
        extractor = FeatureMapping()
        for i in trange(len(dataset)):
            extractor.get(dataset[i])
        return extractor

    @classmethod
    def train(cls, input_file, num_process=16):
        dataset = list(ConllDataset(input_file))
        extractor = FeatureMapping()

        # Splits dataset into subsets for multiprocessing
        step_size = int(len(dataset) / num_process)
        subdataset = [dataset[i:i+step_size] for i in range(0, len(dataset) + 1 - step_size, step_size)]

        # Use multiprocessing to extract features quicker
        # Basically it splits the dataset into processes to extract unknown
        # features faster for the training of the FeatureMapping class
        with Pool(num_process) as pool:
            extractors = pool.map(FeatureMapping.train_on_dataset, subdataset)

            # For each sub feature extractor get all keys
            features = []
            for e in extractors:
                features += list(e.feature.keys())
            
            # Remove all possible duplicate features
            feature_set = set(features)
            extractor.feature = {feature: index for index, feature in enumerate(feature_set, start=1)}
        
        breakpoint()
        return extractor  

    @classmethod
    def save(cls, obj, outfile):
        with gzip.open(outfile,'wb') as stream:
            pickle.dump(obj,stream,-1)

    @classmethod
    def load(cls, infile):
        with gzip.open(infile,'rb') as stream:
            featmap = pickle.load(stream)

        return featmap


if __name__ == '__main__':
    original_file = '../data/wsj_dev.conll06.pred'
    conll_dataset = ConllDataset(original_file)
    feature = FeatureMapping()

    with Pool(16) as pool:
        features = pool.map(feature.get_permutations, list(conll_dataset)[:100])

    # First sanity check
    # 1) Get all indexes from 1 to d-1
    sentence = conll_dataset[0]
    print('Check if indexes 1 to d-1 used and unique')
    val = feature.get(sentence)
    feature_dict = feature.feature

    if None:
        if list(feature_dict[keys[0]].values()) == list(set(feature_dict[keys[0]].values())):
            print('YES')
        else:
            print(feature_dict.values())
            print('NO')


    print('\nTrain on multiple sentences and get final feature')
    feature = FeatureMapping()
    for i in range(100):
        sentence = conll_dataset[i]
        feature.get(sentence)

    sentence = conll_dataset[23]
    feature.frozen = True

    print('Pool')
    pool_dataset = [conll_dataset[i] for i in range(32)]
    with Pool(16) as pool:
        features = pool.map(feature.get_permutations, pool_dataset)
        features = pool.map(feature.get_permutations, pool_dataset)

    pprint(len(features))
