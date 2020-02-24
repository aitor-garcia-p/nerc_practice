"""
The code in this file is only meant to read files in format:

token1 tag1
token2 tag2
...
<empty_line>  <-- sentence boundary
token1 tag1
token2 tag2
..

It contains some ad-hoc changes and may also contain some bugs. It has only been tested for a certain dataset content.
"""

import re


def read_spacy_nerc_instances_from_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    results = load_spacy_train_data_from_bio_dataset(lines)
    return results


def load_spacy_train_data_from_bio_dataset(lines):
    """
    Process dataset (token-tag per line, empty lines as sentence boundary) to obtain lists of tuples in spaCy format
    The spaCy format is something like this:

    ("Who is Shaka Khan?", {"entities": [(7, 17, "PERSON")]}),
    ("I like London and Berlin.", {"entities": [(7, 13, "LOC"), (18, 24, "LOC")]}),

    :param lines: the lines read from a file with suitable training data (token tag per line)
    :return: a list with the training instances converted to the spaCy format
    """

    train_instances = []
    train_instance_text = ''
    train_instance_annotations = []

    current_entity = ''
    current_tag = ''
    current_entity_offset = 0
    for i, line in enumerate(lines):
        if len(line.strip()) == 0 and len(train_instance_text.strip()) > 0:
            train_instances.append((train_instance_text.strip(), {"entities": train_instance_annotations}))
            train_instance_text = ''
            train_instance_annotations = []
            # print("Num train instances:", len(train_instances))
            continue
        if len(line.strip()) == 0 or (len(line.strip()) > 0 and ' ' not in line.strip()):
            # there are double empty lines separating sentences... skip the second...
            # (also there are some lines only with tag, skip them)
            continue

        res = re.sub(r'(.+)\s+([-\w]+)', r'\1###\2', line.strip())
        current_offset = len(train_instance_text)

        token, tag = res.strip().split('###')

        train_instance_text += token + ' '

        # print(line.strip())
        if tag.strip() == '':
            # quick fix to prevent the rare errors in the dataset when a token appears without tag
            tag = 'O'

        if tag.startswith('B'):
            # print("Tag starting with B, current entity:", current_entity)
            # new entity, store previous entity if any
            if len(current_entity.strip()) > 0:
                # there is some previous entity to be stored
                # print("Storing annotation...")
                train_instance_annotations.append((current_entity_offset, current_entity_offset + len(current_entity), current_tag))
            current_entity = token.strip()
            current_tag = tag.split('-')[1].strip()
            current_entity_offset = current_offset

        elif tag.startswith('I'):
            # print("Tag starting with I")
            current_entity += ' ' + token
        elif tag.startswith('O'):
            # print("Tag starting with O")
            if len(current_entity.strip()) > 0:
                # there is some previous entity to be stored
                train_instance_annotations.append(
                    (current_entity_offset, current_entity_offset + len(current_entity), current_tag))
                current_entity = ''
                current_tag = ''
        else:
            raise Exception("ERROR HERE...", tag)
        # print("Current entity:", current_entity, ' current tag:', current_tag)

    train_instances = remove_overlapping_entities(train_instances)
    return train_instances


def remove_overlapping_entities(instances):
    """ Simple strategy of removing each second offending entity"""
    # print('Hi')
    for instance in instances:
        # print(instance)
        entities = instance[1]['entities']
        remaining_entities = []
        for i in range(len(entities)):
            if i >= len(entities) - 1:
                # we are done here, it is the last one
                continue
            ent1 = entities[i]
            ent2 = entities[i + 1]
            if ent1[1] < ent2[0]:
                # we are assuming that the entities are ordered along the sentence
                # under that assumption this condition means no overlapping
                remaining_entities.append(ent1)
                # skip index for ent2
                i += 1
        instance[1]['entities'] = remaining_entities

    return instances


def transform_conll_format_to_plain_text(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    results = []
    current_sentence = []
    for line in lines:
        if line.strip() == '':
            if len(current_sentence) > 0:
                results.append(' '.join(current_sentence))
                current_sentence = []
        else:
            current_sentence.append(line.split()[0])

    # pick the last one
    if len(current_sentence) > 0:
        results.append(' '.join(current_sentence))

    results = [result + '\n' for result in results]
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(results)
