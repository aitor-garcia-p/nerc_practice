import os
import random
import shutil

import spacy
from spacy.util import minibatch
from tqdm import tqdm

from part2_train_custom_nerc.evaluation import evaluate, EvaluationScores


def train_nerc_model(base_lang, train_data, dev_data, output_model_dir, model_name, num_epochs=5):
    """
    Train spaCy NERC model using new train and dev data.
    :param base_lang: the base language for spaCy to instantiate a new blank model
    :param train_data: the training data in spaCy format
    :param dev_data: the development data (for evaluation during training) in spaCy format
    :param output_model_dir: the directory in which you want to store the resulting models
    :param model_name: the name for the model you are training (it will appear in the name of the resulting file)
    :param num_epochs: the number of epochs (full training data loops)
    :return:
    """
    # First we create a new fresh spaCy model instance
    print('Instantiating a fresh model to be trained')
    nlp = _instantiate_model_for_training(base_lang, train_data)

    # This has to do with spaCy: we only want to train NER, so we remove the rest of the "tools" enabled by spaCy
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        # reset and initialize the weights randomly because we're training a new model
        optimizer = nlp.begin_training()
        # create a progress bar, so we can see how the train progresses in the console
        epochs_progress_bar = _epoch_progress_bar(num_epochs=num_epochs)
        # init to zero the best fscore value (the metric we are going to use to measure how "good" the model is)
        best_fscore = 0.0
        # Here is where the training starts, each epoch is a full pass over the training set
        for epoch in epochs_progress_bar:
            # Shuffle the data to increase randomness in each epoch, this helps the learning process
            random.shuffle(train_data)

            # batch up the examples using spaCy's minibatch
            batches = list(minibatch(train_data, size=32))
            # another progress bar, this time for the batches inside an epoch
            with _batch_progress_bar(batches, epoch=epoch, num_epochs=num_epochs) as t:
                # each batch is a group of examples that will be used to perform one "training-step"
                for batch in t:
                    batch_losses = {}
                    # get the text and their gold annotations from the batch
                    texts, annotations = zip(*batch)
                    # this is the training step performed by spaCy, after this the model should have learnt "a tiny bit"
                    # spaCy manages a lot of things behind-the-scenes, we do not need to worry about them
                    nlp.update(
                        texts,  # batch of texts
                        annotations,  # batch of annotations
                        drop=0.5,  # dropout - make it harder to memorise data
                        losses=batch_losses, sgd=optimizer
                    )
                    # we report the "loss" to the progress bar, so we can have an idea about how the training is going
                    __report_to_progress_bar(t, batch_losses['ner'])

            # after a full epoch of training, we evaluate the current status of our model
            scores: EvaluationScores = evaluate(dev_data, nlp)
            # we get the fscore out, because we will focus on it to assess our model (the higher the better)
            current_fscore = scores.fscore
            print('Scores:', [f'{score_name.upper()}:{score_value:1.4f}'
                              for score_name, score_value in scores.list_scores()])

            # compare the previous best fscore with the current one
            # it is better, then we store a new version of our model (we will end up having several versions)
            if current_fscore > best_fscore:
                print(f'New BEST Fscore so far, CURRENT: {current_fscore:1.4f} \
                     PREVIOUS: {best_fscore:1.4f}  DIFF:{current_fscore - best_fscore:1.4f}')
                _save_model(output_model_dir, f'{model_name}_epoch{epoch}_fscore{current_fscore:1.4f}', nlp, optimizer)
                best_fscore = current_fscore


def _save_model(output_model_dir_path, model_name, nlp, optimizer):
    """
    Saves the model to the provided directory, using the provided name
    :param output_model_dir_path: the directory path to store the model
    :param model_name: the name of the model file
    :param nlp: the spaCy model to store
    :param optimizer: the spaCy optimizer used in the training (necessary to correctly store the spaCy model)
    :return:
    """
    if output_model_dir_path is not None:
        if not os.path.exists(output_model_dir_path):
            os.makedirs(os.path.abspath(output_model_dir_path))
        model_path = os.path.join(output_model_dir_path, model_name)
        with nlp.use_params(optimizer.averages):
            print('Saving resulting model...')
            nlp.to_disk(model_path)
            # an additional step to compress the resulting model is necessary
            print('Compressing model to a zip file...')
            shutil.make_archive(model_path, 'zip', model_path)
            print('Deleting uncompressed version of the model...')
            shutil.rmtree(model_path)
        print("Model saved to {}".format(os.path.abspath(model_path + '.zip')))

    else:
        print('No valid output model name was set (non-null and non-existing model name required)')
        raise Exception('No valid output model name was set (non-null and non-existing model name required)')


def _instantiate_model_for_training(base_lang, train_data):
    """
    Instantiate a new blank model so we can train it later
    :param base_lang: the base language to create a base blank model with spaCy
    :param train_data: the training data (we need it here to get all the possible labels from it)
    :return: a blank instance of a spaCy model
    """
    nlp = spacy.blank(base_lang)
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner, last=True)

    # we need an inventory of possible entities to be detected
    # we get them on-the-fly reading them directly from the training data
    entities_set = set()
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            entities_set.add(ent[2])
    for entity in entities_set:
        ner.add_label(entity)
    return nlp


def _epoch_progress_bar(num_epochs):
    """Helper function to clean-up the batch progress bar boilerplate, the parameters are expected to remain constant"""
    t = tqdm(range(num_epochs),
             position=0,
             desc=f'NERC training progress',
             postfix='', ncols=100
             )
    return t


def _batch_progress_bar(batches, epoch, num_epochs):
    """Helper function to clean-up the batch progress bar boilerplate, the parameters are expected to remain constant"""
    t = tqdm(batches, total=len(batches),
             position=0,
             leave=True,
             desc=f'Epoch {epoch}/{num_epochs} progress',
             postfix='', ncols=100
             )
    return t


def __report_to_progress_bar(progress_bar, loss):
    """ Helper function to print the loss in the progress bar"""
    progress_bar.postfix = f'; LOSS:{loss:1.4f}'
