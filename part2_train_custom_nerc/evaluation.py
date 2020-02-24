from dataclasses import dataclass
from typing import Dict


@dataclass
class EvaluationScores:
    """ A helper data class to hold the metrics once they are calculated """
    precision: float
    recall: float
    fscore: float

    def list_scores(self):
        return [('precision', self.precision), ('recall', self.recall), ('fscore', self.fscore)]


def evaluate(test_instances, nlp) -> EvaluationScores:
    """
    Calculate the precision, recall and fscore when using the model to predict the result for test some test instances
    :param test_instances: the instances to evaluate
    :param nlp: the spaCy model to be evaluated
    :return: a instance of the class EvaluationScores, containing the resulting metrics
    """
    # Note: tp, fp and fn mean "True Positives", "False positives" and "False Negatives" respectively
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0
    for text, golds in test_instances:
        predictions = nlp(text).ents
        tp, fp, fn = compare_predictions_and_gold_labels(convert_predictions_to_str_set(predictions),
                                                         convert_gold_to_str_set((text, golds)))
        overall_tp += tp
        overall_fp += fp
        overall_fn += fn

    # PRECISION: tp / tp + fp
    precision = overall_tp / (
            overall_tp + overall_fp) if overall_tp + overall_fp > 0 else 0
    # RECALL: tp / tp + fn
    recall = overall_tp / (
            overall_tp + overall_fn) if overall_tp + overall_fn > 0 else 0
    # FSCORE:  2 * (prec * rec) / (prec + rec)
    fscore = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    # precision,recall and fscore of the named entity recognition
    return EvaluationScores(precision=precision, recall=recall, fscore=fscore)


def convert_predictions_to_str_set(predictions):
    """ Helper method to convert predictions into readable text strings """
    return {x.text + "_" + x.label_ for x in predictions}


def convert_gold_to_str_set(test_instance):
    """ Helper method to convert the gold labels into readable text strings """
    return {test_instance[0][x[0]:x[1]] + '_' + x[2] for x in test_instance[1]['entities']}


def compare_predictions_and_gold_labels(predictions_set, golds_set):
    """ Calculate True Positives (tp), False Positives (fp), and False Negatives (fn) """
    tp, fp, fn = 0, 0, 0
    for pred in predictions_set:
        if pred in golds_set:
            tp += 1
        else:
            fp += 1
    for gold in golds_set:
        if gold not in predictions_set:
            fn += 1
    return tp, fp, fn
