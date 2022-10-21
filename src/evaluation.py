from .dataset import ConllSentence
import numpy as np

def uas(gold, prediction):
    if type(gold).__name__ == 'ConllDataset':
        gold = [gold[i] for i in range(len(gold))]

    if type(prediction).__name__ == 'ConllDataset':
        prediction = [prediction[i] for i in range(len(prediction))]

    match_count = 0
    total_count = 0
    for gold_sentence, prediction_sentence in zip(gold, prediction):
        gold_heads = gold_sentence.sentence['HEAD'].to_numpy()
        prediction_heads = prediction_sentence.sentence['HEAD'].to_numpy()

        count = sum(gold_heads == prediction_heads)
        match_count += count
        total_count += len(gold_heads)

    return match_count / total_count

def las(gold, prediction):
    if type(gold).__name__ == 'ConllDataset':
        gold = [gold[i] for i in range(len(gold))]

    if type(prediction).__name__ == 'ConllDataset':
        prediction = [prediction[i] for i in range(len(prediction))]

    match_count = 0
    total_count = 0
    for gold_sentence, prediction_sentence in zip(gold, prediction):
        gold_val = gold_sentence.sentence[['HEAD', 'REL']].to_numpy()
        pred_val = prediction_sentence.sentence[['HEAD', 'REL']].to_numpy()

        total_count += len(gold_val)
        matches = gold_val == pred_val
        for head, rel in zip(matches[:, 0], matches[:, 1]):
            if head == True and head == rel:
                match_count += 1 


    return match_count / total_count