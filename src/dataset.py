import csv
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

class ConllSentence():
    column_names = ['id', 'form', 'lemma', 'pos', 'xpos', 'morph.', 'head', 'rel', 'deps', 'misc']

    def __init__(self, dataframe):
        self.sentence = dataframe
        self.sentence.set_index('id', inplace=True)

    def set_arcs(self, arcs):
        # Arcs to be sorted 1->N
        arcs.sort(key=lambda x: x[1])

        for arc in arcs:
            head, dep = arc
            self.sentence.loc[dep, 'head']  = head

    def get_arcs(self):
        heads = self.sentence['head'].to_list()
        arcs = [[head, dep] for dep, head in enumerate(heads, start=1)]

        # Sort by dependants
        return sorted(arcs, key=lambda x: x[1])

    def __len__(self):
        return len(self.sentence)

    def __getitem__(self, key):
        item = self.sentence.loc[key]
        return item

    

class ConllDataset():
    ROOT_TOKEN = 0
    column_names = ['id', 'form', 'lemma', 'pos', 'xpos', 'morph.', 'head', 'rel', 'deps', 'misc']
    

    def __init__(self, filepath, delimiter='\t', has_column=False):
        self.has_column_names = has_column

        pandas_args = {
            'filepath_or_buffer': filepath,
            'delimiter': delimiter,
        }

        # When the file doesn't explicitly contain column names
        if not self.has_column_names:
            pandas_args['names'] = self.column_names

        self.dataset = pd.read_csv(**pandas_args)
        self.sentence_indexes = self.dataset[self.dataset.iloc[:, 0] == 1].index.to_list()

    def __getitem__(self, key) -> ConllSentence:
        # Get the start and end index of the sentence in the Dataframe
        begin_index = self.sentence_indexes[key]
        if key >= 0 and len(self) > key + 1:
            # -1 to not already include a token of the next sentence
            end_index = self.sentence_indexes[key + 1]
        else:
            end_index = len(self.dataset)

        sentence_dataframe = self.dataset[begin_index: end_index]
        sentence = ConllSentence(sentence_dataframe)

        return sentence

    def __setitem__(self, key, sentence: ConllSentence):
        # Get the start and end index of the sentence in the Dataframe
        begin_index = self.sentence_indexes[key]
        if key >= 0 and len(self) > key + 1:
            # -1 to not already include a token of the next sentence
            end_index = self.sentence_indexes[key + 1]
        else:
            end_index = len(self.dataset)

        self.dataset[begin_index: end_index].update(sentence.sentence)

    def __len__(self):
        return len(self.sentence_indexes)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def set_arcs(self, sentence_index, arcs):
        sentence = self[sentence_index]
        sentence.set_arcs(arcs)


    def write(self, filepath, add_columns=False):
        self.dataset.to_csv(filepath, sep='\t', index=False, header=add_columns)

        # Adding newlines - this code is horrible
        with open(filepath, 'r') as f:
            contents = f.readlines()

            # Use the sentence indexes from the dictionary
            # Reversed so that we start from the end and ignore first line
            for newline_index in reversed(self.sentence_indexes[1:]):
                contents.insert(newline_index, "\n")

            # They have a newline in the end
            contents.append("\n")

        with open(filepath, 'w') as f:
            f.writelines(contents)

    @staticmethod
    def test_write(dataset, filepath, add_columns=False):
        if type(dataset).__name__ == 'ConllDataset' or type(dataset).__name__ == 'pandas.core.frame.DataFrame':
           dataset.to_csv(filepath, delimiter='\t', index=False, header=add_columns)
        

        

    