from pprint import pprint
import pickle, gzip
from collections import defaultdict
import numpy as np

from dataset import ConllSentence
from dataset import ConllDataset


class FeatureMapping:
    def __init__(self) -> None:
        self.feature = defaultdict(int)
        self.feature_extractors = {
            'hform': self.get_hform,
            'hpos': self.get_hpos,
            'dform': self.get_dform,
            'dpos': self.get_dpos,
            'hform, dpos': self.get_hform_dpos,
            'hpos, dform': self.get_hpos_dform,
        }
        self.frozen = False

    # Feature extractor functions
    # ====
    def get_hform(self, sentence, index_dep, index_head) -> str:
        form = 'ROOT' if index_head == 0 else sentence[index_head]['form']

        return form

    def get_hpos(self, sentence, index_dep, index_head) -> str:
        pos = 'ROOT' if index_head == 0 else sentence[index_head]['pos']

        return pos

    def get_dform(self, sentence, index_dep, index_head) -> str:
        form = sentence[index_dep]['form']
        return form

    def get_dpos(self, sentence, index_dep, index_head) -> str:
        pos = sentence[index_dep]['pos']
        return pos

    def get_hform_dpos(self, sentence, index_dep, index_head) -> str:
        hform = self.get_hform(sentence, index_dep, index_head)
        dpos = self.get_dpos(sentence, index_dep, index_head)

        return f'{hform}+{dpos}'

    def get_hpos_dform(self, sentence, index_dep, index_head) -> str:
        hpos = self.get_hpos(sentence, index_dep, index_head)
        dform = self.get_dform(sentence, index_dep, index_head)

        return f'{hpos}+{dform}'
    # ====

    def feature_count(self) -> int:
        return len(self.feature_extractors)

    def get_feature(self, sentence, index_dep, index_head):
        feature = []
        for name, func in self.feature_extractors.items():
            value = func(sentence, index_dep, index_head)
            name_value = f'{name}: {str(value)}'

            if not self.frozen and name_value not in self.feature:
                # Set index for feature
                self.feature[name_value] += len(self.feature)

            index = self.feature[name_value]
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
    if list(feature_dict.values()) == list(set(feature_dict.values())):
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
    features = feature.get_permutations(sentence)
    print(features.shape)
    pprint(features)