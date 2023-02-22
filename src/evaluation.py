def uas(gold, prediction):
    '''Calculate the unlabeled attachment score

        gold: ndarray (n,m): n sentences with m arcs
        prediction: ndarray (n,m): n sentences with m arcs
    '''
    match_count = 0
    total_count = 0
    for gold_sentence, prediction_sentence in zip(gold, prediction):
        total_count += len(gold_sentence)

        for predicted_arc in prediction_sentence:
            if predicted_arc in gold_sentence:
                match_count += 1

    return (match_count / total_count) * 100

def las(gold, prediction):
    '''Calculate the labeled attachment score

        gold: ndarray (n,m): n sentences with m labels
        prediction: ndarray (n,m): n sentences with m labels
    '''
    match_count = 0
    total_count = 0
    for gold_sentence, prediction_sentence in zip(gold, prediction):
        total_count += len(gold_sentence)

        for predicted_arc in prediction_sentence:
            if predicted_arc in gold_sentence:
                match_count += 1

    return (match_count / total_count) * 100