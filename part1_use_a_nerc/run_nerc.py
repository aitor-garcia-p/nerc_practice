"""
From: https://spacy.io/usage#quickstart
(In Windows)

python -m venv .env
.env\Scripts\activate
pip install -U spacy
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
python -m spacy download es_core_news_sm

Default spaCy NERC models contain these entity types:
https://spacy.io/api/annotation#named-entities
"""
import argparse
import os

import shutil
from collections import Counter
from typing import Dict

import spacy
from spacy import displacy
from spacy.tokens.doc import Doc

# These are the spaCy model for different languages
# spaCy has pre-trained models for more languages, but this is assuming that we have downloaded only: 'en','es','fr'
LANG_MODELS = {'en': 'en_core_web_sm', 'fr': 'fr_core_news_sm', 'es': 'es_core_news_sm'}


def analyze(file_path, language, output_path=None, custom_model=None, top_n=None):
    """
    Analyze an input file in the given language to find out Named Entities
    :param file_path: the file to analyze
    :param language: the language to choose the correct spaCy default model (ignored if custom model is provided)
    :param output_path: the path to write the results to a file (they will be shown in console too)
    :param custom_model: the path to a custom model of your own (if used, the language parameter is ignored)
    :param top_n: the top number of named entities to list in the report that is printed at the end
    :return:
    """
    # Check that the input file exists
    if not os.path.exists(file_path):
        raise Exception(f'Input file does not exist: {file_path}')

    # Load the model (custom or default)
    if custom_model:
        print(f'Loading custom model from {custom_model}')
        nlp = load_custom_model(custom_model)
    else:
        print(f'Loading default spaCy model for {language}')
        nlp = instantiate_default_model(language)

    # Read the input file to analyze it
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Analyze the file
    # It seems pretty simple because spaCy does all the work for us
    # we just need to get the results and do something with them
    # doc is a spaCy Doc object, filled with all the information after the analysis
    # More info about it at the spaCy website: https://spacy.io/api/doc
    doc: Doc = nlp(content)

    # Now we are going to read the detected entities from the doc object, and count them
    entity_type_counter = Counter()
    specific_entities_count_by_type: Dict[str, Counter] = {}
    for ent in doc.ents:
        entity_type_counter.update([ent.label_])
        if ent.label_ in specific_entities_count_by_type:
            specific_entities_count_by_type[ent.label_].update([ent.text])
        else:
            specific_entities_count_by_type[ent.label_] = Counter([ent.text])

    # Prepare some messages with the counts to be printed in the console (or file)
    entities_count_msg = f'==========\nEntity types:\t{entity_type_counter.most_common()}'
    entities_by_type_msg = '==========\nEntities found by type:\n'
    for entity_type, counter in specific_entities_count_by_type.items():
        entities_by_type_msg += f'{entity_type.ljust(10)} =>\t{counter.most_common(top_n)}\n'

    # Print the messages (with our little reports) to console
    print(entities_count_msg)
    print(entities_by_type_msg)

    # if an output_path has been passed as a parameter, write the reports to the path
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(entities_count_msg + '\n')
            f.write(entities_by_type_msg + '\n')

    # Also, using some of the spaCy utility functions, print the detected entities to an HTML page
    html = displacy.render(doc, style="ent", page=True)
    html_file_path = file_path + '._HIGHLIGHTED.html'
    with open(html_file_path, 'w', encoding='utf-8') as f:
        print('>>> NOTE: Writing an HTML file with highlighted entities to: ', html_file_path)
        f.write(html)
        print('>>> The HTML file can be opened with a Web browser (e.g. Firefox, Chrome...)')


def instantiate_default_model(language):
    """ Load an spaCy default model for the given language """
    if language not in LANG_MODELS:
        raise Exception(f'The language {language} is not valid. Use one of: {LANG_MODELS.keys()}')

    model_name = LANG_MODELS[language]
    nlp = spacy.load(model_name)
    return nlp


def load_custom_model(model_path):
    """ Load a custom model from the given path """
    print('Loading custom model: {}'.format(model_path))
    model_zip_path = model_path if model_path.endswith('.zip') else model_path + '.zip'
    shutil.unpack_archive(model_zip_path, extract_dir=model_zip_path[:-4])
    nlp = spacy.load(model_zip_path[:-4])  # remove the.zip extension, and there you should have your path
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    return nlp


def configure_argument_parser():
    """ Console arguments parser configuration """
    parser = argparse.ArgumentParser(description='Perform NERC over a file content.', add_help=True)
    parser.add_argument('--file', type=str, required=True, help='Path to the file to be processed')
    parser.add_argument('--lang', type=str, choices=['en', 'fr', 'es'], default='fr', help='Language of the input')
    parser.add_argument('--output', type=str, required=False, help='Optional path to file to write the results')
    parser.add_argument('--custom_model', type=str, required=False,
                        help='Path to a custom model you have trained or downloaded from elsewhere')
    parser.add_argument('--top_n', type=int, default=10, help='Top N entities to print in the output report')
    return parser


if __name__ == '__main__':
    parser = configure_argument_parser()
    params = parser.parse_args()

    analyze(file_path=params.file, language=params.lang, output_path=params.output, custom_model=params.custom_model,
            top_n=params.top_n)
