from pprint import pprint
from collections import defaultdict
import itertools
import numpy as np
import copy

from dataset import ConllSentence
from dataset import ConllDataset


class FeatureMapping:
    def __init__(self) -> None:
        self.feature = defaultdict(int)
        self.feature_extractors = {
            'hform': self.__get_hform,
            'hpos': self.__get_hpos,
            'dform': self.__get_dform,
            'dpos': self.__get_dpos,
            'hform, dpos': self.__get_hform_dpos,
            'hpos, dform': self.__get_hpos_dform,
        }
        self.frozen = False

    # Feature extractor functions
    # ====
    def __get_hform(self, sentence, index_dep, index_head) -> str:
        form = 'ROOT' if index_head == 0 else sentence[index_head]['form']

        return form

    def __get_hpos(self, sentence, index_dep, index_head) -> str:
        pos = 'ROOT' if index_head == 0 else sentence[index_head]['pos']

        return pos

    def __get_dform(self, sentence, index_dep, index_head) -> str:
        form = sentence[index_dep]['form']
        return form

    def __get_dpos(self, sentence, index_dep, index_head) -> str:
        pos = sentence[index_dep]['pos']
        return pos

    def __get_hform_dpos(self, sentence, index_dep, index_head) -> str:
        hform = self.__get_hform(sentence, index_dep, index_head)
        dpos = self.__get_dpos(sentence, index_dep, index_head)

        return f'{hform}+{dpos}'

    def __get_hpos_dform(self, sentence, index_dep, index_head) -> str:
        hpos = self.__get_hpos(sentence, index_dep, index_head)
        dform = self.__get_dform(sentence, index_dep, index_head)

        return f'{hpos}+{dform}'
    # ====

    def feature_count(self) -> int:
        return len(self.feature_extractors)

    def get(self, sentence: ConllSentence) -> np.ndarray:
        features = []
        # indexes should be dependants (so look at who the head is)
        for index_dep in range(1, len(sentence) + 1):
            feature = []
            for name, func in self.feature_extractors.items():
                index_head = sentence[index_dep]['head']

                # Indexes are set to 1
                value = func(sentence, index_dep, index_head)
                name_value = f'{name}: {str(value)}'

                if not self.frozen and name_value not in self.feature:
                    # Set index for feature
                    self.feature[name_value] += len(self.feature)

                index = self.feature[name_value]
                feature.append(index)

            features.append(feature)

        return np.array(features)

    def get_permutations(self, sentence: ConllSentence) -> np.ndarray:
        '''Used for getting all possible arc features'''
        features = []

        for index_head in range(0, len(sentence) + 1):
            feature = [np.zeros(self.feature_count(),dtype=int)]
            for index_dep in range(1, len(sentence) + 1):
                f = np.zeros(self.feature_count(),dtype=int)

                if index_head != index_dep:
                    for i, (name, func) in enumerate(self.feature_extractors.items()):
                        value = func(sentence, index_dep, index_head)
                        name_value = f'{name}: {str(value)}'

                        f[i] = self.feature[name_value]
                
                feature.append(f)

            features.append(feature)

        return np.asarray(features, dtype=object)





if __name__ == '__main__':
    original_file = '../data/wsj_dev.conll06.pred'
    conll_dataset = ConllDataset(original_file)
    feature = FeatureMapping()

    # First sanity check
    # 1) Get all indexes from 1 to d-1
    sentence = conll_dataset[0]
    print('Check if indexes 1 to d-1 used and unique')
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