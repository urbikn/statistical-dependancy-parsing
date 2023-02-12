import argparse
import pickle
import gzip
from tqdm import trange, tqdm
from multiprocessing import Pool

from src.dataset import ConllDataset
from src.features import FeatureMapping
from src.model import AveragePerceptron
from src import evaluation

# Utility function to help read big Pickle files
def read_from_pickle(path):
    with open(path, 'rb') as file:
        try:
            while True:
                yield pickle.load(file)
        except EOFError:
            pass


def create_dataset(input_file, extractor, output_file, type='training', num_process=32):
    dataset = ConllDataset(input_file)

    batch = []
    train_dataset = []
    items_per_process = 32
    for i, instance in tqdm(enumerate(dataset, start=1), total=len(dataset), desc=f'Extracting features from {type} dataset'):
        batch.append(instance)
        # This small trick because for each process we have to copy the extractor
        if i % (num_process * items_per_process)  == 0 or i == len(dataset):
            with Pool(num_process) as pool:
                output = pool.map(extractor.get_permutations, batch, items_per_process)
                
                for l in range(len(batch)):
                    train_dataset.append([
                        output[l],
                        batch[l].get_arcs()
                    ])

            with open(output_file,'ab') as stream:
                pickle.dump(train_dataset,stream)
            batch = []
            train_dataset = []

    return train_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-feature-file', type=str)
    parser.add_argument('--extractor', type=str)
    parser.add_argument('--save-extractor', type=str, default='./extractor.p')

    parser.add_argument('--process-data', type=bool, default=False)
    parser.add_argument('--train-input', type=str)
    parser.add_argument('--train-dataset', type=str)
    parser.add_argument('--dev-input', type=str)
    parser.add_argument('--dev-dataset', type=str)
    parser.add_argument('--test-input', type=str)
    parser.add_argument('--test-dataset', type=str)

    parser.add_argument('--weights', type=str)
    parser.add_argument('--save-model', type=str, default=None)
    parser.add_argument('--num_process', type=int, default=0)
    args = parser.parse_args()

    # ==== Feature extractor ====
    if args.train_feature_file:
        print('== Training feature extractor ==')
        feature_extractor = FeatureMapping.train(args.train_feature_file)
        FeatureMapping.save(feature_extractor, args.save_extractor)
    if args.extractor:
        feature_extractor = FeatureMapping.load(args.extractor)
    feature_extractor.frozen = True

    # ==== Perceptron ====
    if args.weights:
        model = AveragePerceptron.load(args.weights)
    else:
        model = AveragePerceptron(feature_extractor, dim=len(feature_extractor) + 1)

    # == Processing input files ==
    if args.process_data:
        if args.train_input:
            create_dataset(args.train_input, feature_extractor, args.train_dataset, 'training', args.num_process)
        if args.dev_input:
            create_dataset(args.dev_input, feature_extractor, args.dev_dataset, 'development', args.num_process)
        if args.test_input:
            pass
            # test_dataset = get_test_dataset(args.test_dataset, feature_extractor, 32)
            test_dataset = None

            with gzip.open(args.test_dataset,'wb') as stream:
                pickle.dump(test_dataset,stream,-1)

    # == Processing training dataset ==
    if args.train_dataset:
        train_dataset_list = list(read_from_pickle(args.train_dataset))
        train_dataset = [item for dataset_list in train_dataset_list for item in dataset_list]

    dev_dataset = None
    if args.dev_dataset:
        dev_dataset_list = list(read_from_pickle(args.dev_dataset))
        dev_dataset = [item for dataset_list in dev_dataset_list for item in dataset_list]
        
    if args.test_dataset:
        with open(args.test_dataset,'rb') as stream:
            dataset = pickle.load(stream)
            test_dataset = feature_extractor.features_to_tensors(dataset)


    model.train(train_dataset, dev_dataset, epoch=40, eval_interval=500, learning_rate=1, save_folder=args.save_model)

