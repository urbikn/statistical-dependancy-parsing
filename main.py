import argparse
import pickle
import gzip
from tqdm import trange, tqdm
from multiprocessing import Pool


from src.dataset import ConllDataset
from src.features import FeatureMapping
from src.model import AveragePerceptron

def train_feature_extractor(input_file, output_file):
    dataset = ConllDataset(input_file)
    feature_extractor = FeatureMapping()

    print('== Training feature extractor ==')
    for i in trange(len(dataset)):
        sentence = dataset[i]
        feature_extractor.get(sentence)

    print('Saving feature extractor to:', output_file)
    FeatureMapping.save(feature_extractor, output_file)

def get_train_dataset_multiprocess(dataset, extractor, multiprocess=32):
    train_dataset = []
    batch = []
    for i, instance in tqdm(enumerate(dataset, start=1), total=len(dataset)):
        batch.append(instance)

        if i % multiprocess == 0 or i == len(dataset):
            with Pool(multiprocess) as pool:
                output = pool.map(extractor.get_permutations, batch)
                output_true = pool.map(extractor.get, batch)
                
            for l in range(multiprocess):
                train_dataset.append([
                    output[l],
                    output_true[l],
                    batch[l].get_arcs()
                ])
            batch = []
    return train_dataset


def get_train_dataset(input_file, extractor, multiprocess=0):
    dataset = ConllDataset(input_file)
    extractor.frozen = True
    train_dataset = []
    print('Extracting features from training dataset')
    if multiprocess != 0:
        train_dataset = get_train_dataset_multiprocess(dataset, extractor, multiprocess)
    else:
        for i in trange(len(dataset)):
            sentence = dataset[i]

            x = extractor.get_permutations(sentence)
            x_true = extractor.get(sentence)
            arcs = dataset[i].get_arcs()
            train_dataset.append((x, x_true, arcs))


    return train_dataset

def get_test_dataset(input_file, extractor, multiprocess=0):
    dataset = ConllDataset(input_file)
    extractor.frozen = True
    train_dataset = []
    print('Extracting features from training dataset')
    if multiprocess != 0:
        train_dataset = get_train_dataset_multiprocess(dataset, extractor, multiprocess)
    else:
        for i in trange(len(dataset)):
            sentence = dataset[i]

            x = extractor.get_permutations(sentence)
            x_true = extractor.get(sentence)
            arcs = dataset[i].get_arcs()
            train_dataset.append((x, x_true, arcs))


    return train_dataset



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-feature-file', type=str)
    parser.add_argument('--extractor', type=str)
    parser.add_argument('--save-extractor', type=str, default='./extractor.p')

    parser.add_argument('--train-dataset', type=str)
    parser.add_argument('--train-dataset-features', type=str)
    parser.add_argument('--save-train-dataset', type=str, default='./dataset.p')
    parser.add_argument('--dev-dataset', type=str)

    parser.add_argument('--test-dataset', type=str)
    parser.add_argument('--test-dataset-features', type=str)
    parser.add_argument('--save-test-dataset', type=str, default='./dataset_test.p')

    parser.add_argument('--features', type=str)
    parser.add_argument('--weights', type=str)
    args = parser.parse_args()

    # ==== Feature extractor ====
    if args.train_feature_file:
        train_feature_extractor(args.train_feature_file, args.save_extractor)
    if args.extractor:
        feature_extractor = FeatureMapping.load(args.extractor)
    feature_extractor.frozen = True

    # ==== Perceptron ====
    if args.weights:
        model = AveragePerceptron.load(args.weights)
    else:
        model = AveragePerceptron(dim=feature_extractor.feature_count())

    # ==== Training dataset ====
    if args.train_dataset:
        train_dataset = get_train_dataset(args.train_dataset, feature_extractor, 64)

        if args.save_train_dataset:
            with gzip.open(args.save_train_dataset,'wb') as stream:
                pickle.dump(train_dataset,stream,-1)
    elif args.train_dataset_features:
        with gzip.open(args.train_dataset_features,'rb') as stream:
            train_dataset = pickle.load(stream)

    # ==== Test dataset ====
    if args.test_dataset:
        test_dataset = get_test_dataset(args.test_dataset, feature_extractor, 64)

        if args.save_test_dataset:
            with gzip.open(args.save_test_dataset,'wb') as stream:
                pickle.dump(test_dataset,stream,-1)
    elif args.test_dataset_features:
        with gzip.open(args.test_dataset_features,'rb') as stream:
            test_dataset = pickle.load(stream)


    model.train(train_dataset, epoch=10)

    