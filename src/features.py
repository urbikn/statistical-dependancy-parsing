from pprint import pprint
import pickle, gzip
from collections import defaultdict
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
        }

        self.feature = {name:defaultdict(int) for name in self.feature_extractors.keys()}
        self.frozen = False

    # Feature extractor functions
    # ====
    def hform(self, sentence, index_dep, index_head) -> str:
        form = 'ROOT' if index_head == 0 else sentence[index_head]['form']

        return form

    def hpos(self, sentence, index_dep, index_head) -> str:
        pos = 'ROOT' if index_head == 0 else sentence[index_head]['pos']

        return pos

    def dform(self, sentence, index_dep, index_head) -> str:
        form = sentence[index_dep]['form']
        return form

    def dpos(self, sentence, index_dep, index_head) -> str:
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
    # ====

    def feature_count(self) -> int:
        return len(self.feature_extractors)

    def get_feature(self, sentence, index_dep, index_head):
        feature = []
        for name, func in self.feature_extractors.items():
            value = func(sentence, index_dep, index_head)
            name_value = f'{name}: {str(value)}'

            if not self.frozen and name_value not in self.feature[name]:
                # Set index for feature
                self.feature[name][name_value] += len(self.feature[name])

            index = self.feature[name][name_value]
            feature.append(index)

        return feature
        

    def get(self, sentence: ConllSentence) -> np.ndarray:
        # indexes should be dependants (so look at who the head is)
        features = []
        for index_dep in range(1, len(sentence) + 1):
            index_head = sentence[index_dep]['head']

            feature = self.get_feature(sentence, index_dep, index_head)
            features.append(feature)

        return np.asarray(features, dtype=object)

    def get_permutations(self, sentence: ConllSentence, default=0) -> np.ndarray:
        '''Used for getting all possible arc features. Don't add new features'''
        features = []

        # save variable to see if we should unfreeze
        unfreeze = not self.frozen
        # Freeze feature extractor
        self.frozen = True

        for index_head in range(0, len(sentence) + 1):
            # Sets the first feature to the default value (to denote all the )
            feature = [np.full(self.feature_count(), default)]
            for index_dep in range(1, len(sentence) + 1):
                if index_head != index_dep:
                    f = self.get_feature(sentence, index_dep, index_head)
                else:
                    f = np.full(self.feature_count(), default)
                
                feature.append(f)

            features.append(feature)

        if unfreeze:
            self.frozen = False

        return np.asarray(features, dtype=object)

    @classmethod
    def save(cls, obj, outfile):
        with gzip.open(outfile,'wb') as stream:
            pickle.dump(obj,stream,-1)

    @classmethod
    def load(cls, infile):
        with gzip.open(infile,'rb') as stream:
            featmap = pickle.load(stream)

        return featmap

from multiprocessing import Process
from multiprocessing import Pool



if __name__ == '__main__':
    original_file = '../data/wsj_dev.conll06.pred'
    conll_dataset = ConllDataset(original_file)
    feature = FeatureMapping()

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
    pool_dataset = [conll_dataset[i] for i in range(10)]
    with Pool(16) as pool:
        features = pool.map(feature.get_permutations, pool_dataset)

    pprint(len(features))