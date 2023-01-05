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
        }
        self.frozen = False

    # Feature extractor functions
    def __get_hform(self, sentence, index) -> str:
        head_index = sentence.iloc[index]['head'] - 1

        if head_index >= 0:
            form = sentence.iloc[head_index]['form']
        else:
            form = 'ROOT'

        return form

    def __get_hpos(self, sentence, index) -> str:
        head_index = sentence.iloc[index]['head'] - 1
        if head_index >= 0:
            pos = sentence.iloc[head_index]['pos']
        else:
            pos = 'ROOT'

        return pos

    def __get_dform(self, sentence, index) -> str:
        form = sentence.iloc[index]['form']
        return form

    def __get_dpos(self, sentence, index) -> str:
        pos = sentence.iloc[index]['pos']
        return pos
    

    def get(self, sentence: ConllSentence) -> np.ndarray:
        features = []
        # indexes should be dependants (so look at who the head is)
        for i, row in sentence.iterrows():
            feature = []
            for name, extractor_funct in self.feature_extractors.items():
                val = extractor_funct(sentence, i)
                feature.append(f'{name}: {str(val)}')

            features.append(feature)

        return features

if __name__ == '__main__':
    original_file = '../data/wsj_dev.conll06.pred'
    conll_dataset = ConllDataset(original_file)
    sentence = conll_dataset[0].sentence

    feature = FeatureMapping()

    pprint(feature.get(sentence))