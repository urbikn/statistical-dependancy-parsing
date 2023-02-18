from pprint import pprint
import pickle, gzip
from collections import defaultdict
from multiprocessing import Pool
from tqdm.auto import tqdm, trange
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel

from src.dataset import ConllSentence
from src.dataset import ConllDataset

class BertMapping:
    def __init__(self, model) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model, output_hidden_states=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.eval()
        self.model.to(self.device)

    def get_features(self, sentence):
        sentence= sentence.strip() # Remove trailing characters
        line = '[CLS] ' + line + ' [SEP]'

        tokenized_text = self.tokenizer.tokenize(sentence, is_split_into_words=True)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens]).to(self.device)

        with torch.no_grad():
            output = self.model(tokens_tensor)
            top_hidden_states = output.last_hidden_state
        
        return top_hidden_states


    def create(self, dataset_file, output_file, type='training'):
        dataset = ConllDataset(dataset_file)
        save_dataset = []
        for i, instance in tqdm(enumerate(dataset, start=1), total=len(dataset), desc=f'Extracting features from {type} dataset'):
            output = self.get_features(instance)
            save_dataset.append([output, instance.get_arcs()])

        with gzip.open(output_file,'ab') as stream:
            pickle.dump(save_dataset, stream)


    def load(self, input_file):
        pass



class FeatureMapping:
    def __init__(self) -> None:
        self.feature_extractors = {
            'hshape': self.hshape,
            'dshape': self.dshape,
            'hform': self.hform,
            'hpos': self.hpos,
            'dform': self.dform,
            'dpos': self.dpos,
            'hform, hpos': self.hform_pos,
            'dform, dpos': self.dform_pos,
            'hform, dform': self.hform_dform,
            'hlemma, dlemma': self.hlemma_dlemma,
            'hlemma, dpos': self.hlemma_dpos,
            'hpos, dlemma': self.hpos_dlemma,
            'hform, dpos': self.hform_dpos,
            'hpos, dform': self.hpos_dform,
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
            'hpos between dpos': self.between_pos,
            'hpos between per dpos': self.between_pos_per
        }

        self.frozen = False 
        self.feature = defaultdict(int)

    # Feature extractor functions
    # ====
    def distance(self, sentence, index_dep, index_head) -> str:
        distance = abs(index_dep - index_head)
        distance = distance if distance <= 4 else '+4'

        return f'{distance}'

    def direction(self, sentence, index_dep, index_head) -> str:
        if index_dep < index_head:
            return 'left'
        else:
            return 'right'
    
    def shape(self, term):
        shape = ''
        for char in term:
            if char.isupper():
                shape += 'X'
            elif char.islower():
                shape += 'x'
            elif char.isdigit():
                shape += 'd'
            elif char in ('(', ')', '[', ']', '{', '}'):
                shape += 'b'
            elif char in (',', '.', ';', ':', '?', '!', "'", '"'):
                shape += char
            else:
                shape += 'c'
        return shape

    def hshape(self, sentence, index_dep, index_head) -> str:
        form = self.hform(sentence, index_dep, index_head)

        if form != '__NULL__' and form != 'ROOT':
            return self.shape(form)
        else:
            form

    def hform(self, sentence, index_dep, index_head) -> str:
        if index_head < 0 or len(sentence) < index_head:
            form = '__NULL__'
        elif index_head == 0:
            form = 'ROOT'
        else:
            form = sentence[index_head]['form']

        return form

    def hlemma(self, sentence, index_dep, index_head) -> str:
        if index_head < 0 or len(sentence) < index_head:
            form = '__NULL__'
        elif index_head == 0:
            form = 'ROOT'
        else:
            form = sentence[index_head]['lemma']

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

    def dshape(self, sentence, index_dep, index_head) -> str:
        form = self.dform(sentence, index_dep, index_head)

        if form != '__NULL__':
            return self.shape(form)
        else:
            form
    
    

    def dlemma(self, sentence, index_dep, index_head) -> str:
        if index_dep <= 0 or len(sentence) < index_dep:
            form = '__NULL__'
        else:
            form = sentence[index_dep]['lemma']

        return form

    def dpos(self, sentence, index_dep, index_head) -> str:
        if index_dep <= 0 or len(sentence) < index_dep:
            pos = '__NULL__'
        else:
            pos = sentence[index_dep]['pos']

        return pos

    # HELPER
    def hform_pos(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.hform(*params)} {self.hpos(*params)}'

    # HELPER
    def dform_pos(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.dform(*params)} {self.dpos(*params)}'

    def hform_dpos(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.hform(*params)} {self.dpos(*params)}'

    def hpos_dform(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.hpos(*params)} {self.dform(*params)}'

    def hform_dform(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.hform(*params)} {self.dform(*params)}'

    def hlemma_dlemma(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.hlemma(*params)} {self.dlemma(*params)}'

    def hlemma_dpos(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.hlemma(*params)} {self.dpos(*params)}'

    def hpos_dlemma(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.hpos(*params)} {self.dlemma(*params)}'

    def hpos_dpos(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.hpos(*params)} {self.dpos(*params)}'

    def hformpos_dformpos(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.hform_pos(*params)} {self.dform_pos(*params)}'

    def hformpos_dform(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.hform_pos(*params)} {self.dform(*params)}'

    def hformpos_dpos(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.hform_pos(*params)} {self.dpos(*params)}'

    def hform_dformpos(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.hform(*params)} {self.dform_pos(*params)}'

    def hpos_dformpos(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        return f'{self.hpos(*params)} {self.dform_pos(*params)}'

    def hpos_next_dpos_prev(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        params_neighbors = (sentence, index_dep - 1, index_head + 1)
        return f'{self.hpos(*params)} {self.hpos(*params_neighbors)} {self.dpos(*params)} {self.dpos(*params_neighbors)}'
        
    def hpos_prev_dpos_next(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        params_neighbors = (sentence, index_dep+1, index_head-1)
        return f'{self.hpos(*params)} {self.hpos(*params_neighbors)} {self.dpos(*params)} {self.dpos(*params_neighbors)}'

    def hpos_next_dpos_next(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        params_neighbors = (sentence, index_dep+1, index_head+1)
        return f'{self.hpos(*params)} {self.hpos(*params_neighbors)} {self.dpos(*params)} {self.dpos(*params_neighbors)}'

    def hpos_prev_dpos_prev(self, sentence, index_dep, index_head) -> str:
        params = (sentence, index_dep, index_head)
        params_neighbors = (sentence, index_dep-1, index_head-1)
        return f'{self.hpos(*params)} {self.hpos(*params_neighbors)} {self.dpos(*params)} {self.dpos(*params_neighbors)}'
    

    def between_pos(self, sentence, index_dep, index_head) -> str:
        if index_dep < index_head:
            val = " ".join([self.hpos(sentence, 0, i) for i in range(index_dep+1, index_head)])
        else:
            val = " ".join([self.hpos(sentence, 0, i) for i in range(index_head+1, index_dep)])

        return val if len(val) else "__NULL__"

    def between_pos_per(self, sentence, index_dep, index_head) -> list:
        start, end = (index_head, index_dep) if index_head < index_dep else (index_dep, index_head)

        features = []
        for i in range(start+1, end):
            between_pos = self.dpos(sentence, i, 0)
            features.append(f'between pos: {between_pos}')

        if len(features) == 0:
            features = ['between pos: __NULL__']
        
        return features

    # ====

    def __len__(self) -> int:
        return len(self.feature)

    def num_features(self) -> int:
        return len(self.feature_extractors)

    def get_feature(self, sentence, index_dep, index_head):
        '''Get index (dictionary key storing index) of features for the head->dependant given a sentence.'''

        feature = []
        for i, (name, func) in enumerate(self.feature_extractors.items()):
            if name == 'hpos between per dpos':
                continue

            value = str(func(sentence, index_dep, index_head)) 
            value += f' {self.distance(sentence, index_dep, index_head)} {self.direction(sentence, index_dep, index_head)}'
            name_value = f'{name}: {str(value)}'

            if not self.frozen and name_value not in self.feature:
                # Set index for feature
                self.feature[name_value] += len(self.feature)

            feature.append(self.feature.get(name_value, 0))

        for value_name in self.between_pos_per(sentence, index_dep, index_head):
            value_name += f' {self.distance(sentence, index_dep, index_head)} {self.direction(sentence, index_dep, index_head)}'

            if not self.frozen and value_name not in self.feature:
                # Set index for feature
                self.feature[value_name] += len(self.feature)

            feature.append(self.feature.get(value_name, 0))

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

    def get_permutations(self, sentence: ConllSentence, default=0, freeze=True):
        '''Used for getting all possible arc features. Don't add new features'''
        # save variable to see if we should unfreeze
        # Freeze feature extractor

        features = []
        for index_head in range(0, len(sentence) + 1):
            # Sets the first feature to the default value (to denote all the )
            feature = [[0]]
            for index_dep in range(1, len(sentence) + 1):
                if index_head != index_dep:
                    f = self.get_feature(sentence, index_dep, index_head)
                else:
                    f = [0]
                
                feature.append(f)

            features.append(feature)

        return features

        
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
        
        return extractor  

    @classmethod
    def save(cls, obj, outfile):
        with gzip.open(outfile,'wb') as stream:
            pickle.dump(obj,stream,-1)

    @classmethod
    def load(cls, infile):
        print('Read extractor from:', infile)
        with gzip.open(infile,'rb') as stream:
            featmap = pickle.load(stream)

        print('Finished reading extractor.')
        return featmap


if __name__ == '__main__':
    original_file = '../data/wsj_dev.conll06.pred'
    conll_dataset = ConllDataset(original_file)
    feature = FeatureMapping()

    #with Pool(16) as pool:
    #    features = pool.map(feature.get_permutations, list(conll_dataset)[:100])

    # First sanity check
    # 1) Get all indexes from 1 to d-1
    sentence = conll_dataset[10]
    print('Check if indexes 1 to d-1 used and unique')
    val = feature.get_permutations(sentence)
    breakpoint()
    feature_dict = feature.feature

    print(" ".join([sentence[i]['form'] for i in range(1, len(sentence)+1)]))
    pprint({v:k for k, v in feature_dict.items()})

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
