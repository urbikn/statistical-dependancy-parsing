from pprint import pprint
from collections import defaultdict
import numpy as np

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

        # TODO: You can also try using just feature indexes
        # Used during training as next index for new added features
        # self.feature_current_index = {feature_name:1 for feature_name in self.feature_extractors.keys()}

    # Feature extractor functions
    # ====
    def __get_hform(self, sentence, index) -> str:
        try:
            head_index = sentence[index]['head']
        except:
            print(head_index)
        form = 'ROOT' if head_index == 0 else sentence[head_index]['form']

        return form

    def __get_hpos(self, sentence, index) -> str:
        head_index = sentence[index]['head']
        pos = 'ROOT' if head_index == 0 else sentence[head_index]['pos']

        return pos

    def __get_dform(self, sentence, index) -> str:
        form = sentence[index]['form']
        return form

    def __get_dpos(self, sentence, index) -> str:
        pos = sentence[index]['pos']
        return pos

    def __get_hform_dpos(self, sentence, index) -> str:
        hform = self.__get_hform(sentence, index)
        dpos = self.__get_dpos(sentence, index)

        return f'{hform}+{dpos}'

    def __get_hpos_dform(self, sentence, index) -> str:
        hpos = self.__get_hpos(sentence, index)
        dform = self.__get_dform(sentence, index)

        return f'{hpos}+{dform}'
    # ====


    def get(self, sentence: ConllSentence) -> np.ndarray:
        features = []
        # indexes should be dependants (so look at who the head is)
        for i in range(len(sentence)):
            feature = []
            for name, func in self.feature_extractors.items():
                # Indexes are set to 1
                value = func(sentence, i + 1)
                name_value = f'{name}: {str(value)}'

                if not self.frozen and name_value not in self.feature:
                    # Set index for feature
                    self.feature[name_value] += len(self.feature)
                    # incremenet tempalate feature index
                    #self.feature[name_value] = self.feature_current_index[name]
                    #self.feature_current_index[name] += 1

                index = self.feature[name_value]
                feature.append(index)

            features.append(feature)

        return features

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

    sentence = conll_dataset[100]
    feature.frozen
    pprint(feature.get(sentence))