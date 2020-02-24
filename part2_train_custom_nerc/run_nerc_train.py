import argparse
import os

from part2_train_custom_nerc.data_conversion import read_spacy_nerc_instances_from_file
from part2_train_custom_nerc.spacy_nerc_training import train_nerc_model


def train(train_set_path, dev_set_path, output_model_dir, model_name, base_language='en', num_epochs='5'):
    """
    Reads the train/dev data from their respective locations and launches a NERC model training process
    :param train_set_path: path to the training set file in the correct format
    :param dev_set_path: path to the development set file in the correct format
    :param output_model_dir: directory to store the resulting model versions
    :param model_name: name of the model to be trained (will be used as the name of the resulting files)
    :param base_language: the base language for the (spaCy) blank model that will be trained
    :param num_epochs: number of epochs (full training loops)
    :return:
    """
    if not _check_path_exist(train_set_path):
        print(f'The TRAIN set path DOES NOT EXIST, please check it: {os.path.abspath(train_set_path)}')
        return
    if not _check_path_exist(dev_set_path):
        print(f'The DEV set path DOES NOT EXIST, please check it: {os.path.abspath(dev_set_path)}')
        return
    print('Converting training input data to a format suitable for training...')
    train_instances = read_spacy_nerc_instances_from_file(train_set_path)
    print('Converting evaluation input data to a format suitable for training...')
    dev_instances = read_spacy_nerc_instances_from_file(dev_set_path)
    train_nerc_model(base_lang=base_language, train_data=train_instances, dev_data=dev_instances,
                     output_model_dir=output_model_dir,
                     model_name=model_name, num_epochs=num_epochs)

    print(f'Training stopped after {num_epochs} epochs')


def _check_path_exist(path):
    """ Returns true if the given path exists"""
    return os.path.exists(path)


def configure_argument_parser():
    """ Console arguments parser configuration """
    parser = argparse.ArgumentParser(description='Perform NERC over a file content.', add_help=True)
    parser.add_argument('--train_data', type=str, required=True, help='Path to the training data')
    parser.add_argument('--dev_data', type=str, required=True, help='Path to the development data for evaluation')
    parser.add_argument('--lang', type=str, choices=['en', 'fr', 'es'], default='en',
                        help='Base language to train the model from (to create an spaCy blank model)')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs (full training set loops)')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the folder to store the trained models')
    parser.add_argument('--model_name', type=str, required=False, default='nerc_model',
                        help='Name for the model to be trained (will be used as part of the name of the stored model')
    return parser


if __name__ == '__main__':
    parser = configure_argument_parser()
    params = parser.parse_args()

    train(train_set_path=params.train_data, dev_set_path=params.dev_data,
          output_model_dir=params.output_dir, model_name=params.model_name,
          base_language=params.lang, num_epochs=params.num_epochs)
