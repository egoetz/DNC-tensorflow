# -*- coding: utf-8 -*-

import sys
import pickle
import getopt
import json
import numpy as np
import re
from shutil import rmtree
from os import listdir, mkdir
from os.path import join, isfile, isdir, dirname, basename, normpath, abspath, exists
from word2number import w2n
from string import punctuation
from cleaning import sentence_dict, spacing_dict, pattern_dict, spelling_dict


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def clean_sentences(sentence_list):
    should_print = [[], [], []]
    new_sentence_list = []
    for sentence in sentence_list:
        if sentence in sentence_dict.keys():
            sentence_list[sentence_list.index(sentence)] = sentence_dict[sentence]
    for index, sentence in enumerate(sentence_list):
        # first separate . and ? away from words into separate lexicons
        capitalized = set()
        abbrivations = set()
        for word in sentence.split():
            if word.istitle():
                capitalized.add(word)
            if word.isupper():
                abbrivations.add(word)
        new_sentence = sentence.split(" ")
        for i in range(len(new_sentence)):
            if new_sentence[i] in capitalized:
                new_sentence[i] = f" ^ {new_sentence[i]}"
            if new_sentence[i] in abbrivations:
                new_sentence[i] = f" ^^ {new_sentence[i]}"
        new_sentence = " ".join(new_sentence)
        new_sentence = new_sentence.lower()
        for key in spacing_dict.keys():
            old_sentence = new_sentence
            new_sentence = new_sentence.replace(key, spacing_dict[key])
            if old_sentence != new_sentence:
                should_print[0].append(f"Spacing replacement, key {key}")
        for key in spelling_dict.keys():
            old_sentence = new_sentence
            new_sentence = new_sentence.replace(key, spelling_dict[key])
            if old_sentence != new_sentence:
                should_print[1].append(f"Spelling replacement, key {key}")
        for word in sentence:
            try:
                new_word = w2n.word_to_num(word)
                new_sentence = new_sentence.replace(word, str(new_word))
            except ValueError:
                continue
        new_sentence = new_sentence.replace('-', ' - ')
        words = new_sentence.split(" ")
        for word in words:
            for pattern in pattern_dict.keys():
                match = re.fullmatch(pattern, word)
                if match:
                    old_sentence = new_sentence
                    new_sentence = old_sentence.split()
                    for i in range(len(new_sentence)):
                        if new_sentence[i] == word:
                            new_sentence[i] = pattern_dict[pattern]
                    new_sentence = " ".join(new_sentence)
                    if old_sentence != new_sentence:
                        should_print[2].append(f"Regex replacement, pattern {pattern}")

        words = new_sentence.split()
        for word in words:
            if word.isnumeric():
                new_str = ""
                length_num = len(word)
                for char in word:
                    if len(new_str) != 0:
                        new_str += f" {char}"
                    else:
                        new_str = char
                old_sentence = new_sentence
                new_sentence = old_sentence.split()
                for i in range(len(new_sentence)):
                    if new_sentence[i] == word:
                        new_sentence[i] = new_str
                new_sentence = " ".join(new_sentence)
        new_sentence_list.append(new_sentence)
    return new_sentence_list


def clean_data(data):
    for i, entry in enumerate(data):
        for j in range(len(entry)):
            if j == 0:
                data[i][j] = clean_sentences(entry[0])
            elif j == 1:
                for k, question_dictionary in enumerate(entry[j]):
                    data[i][j][k]["question"] = clean_sentences([question_dictionary["question"]])[0]
                    data[i][j][k]["answer"] = clean_sentences([question_dictionary["answer"]])[0]

    return data


def create_dictionary(data):
    """
    creates a dictionary of unique lexicons in the dataset and their mapping to numbers

    Parameters:
    ----------
    files_list: list
        the list of files to scan through

    Returns: dict
        the constructed dictionary of lexicons
    """

    lexicons_dict = {}
    id_counter = 0

    llprint("Creating Dictionary ... 0/%d" % (len(data)))

    for index, entry in enumerate(data):
        sentences = entry[0]
        for question_dictionary in entry[1]:
            sentences.append(question_dictionary["question"])
            sentences.append(question_dictionary["answer"])
        for sentence in sentences:
            for word in sentence.split():
                if not word.lower() in lexicons_dict:
                    lexicons_dict[word.lower()] = id_counter
                    id_counter += 1

        llprint("\rCreating Dictionary ... %d/%d" % ((index + 1), len(data)))
    lexicons_dict['+'] = id_counter
    id_counter += 1
    lexicons_dict['\\'] = id_counter
    id_counter += 1
    lexicons_dict['='] = id_counter

    print("\rCreating Dictionary ... Done!")
    return lexicons_dict


def encode_sentences(sentences, lexicon_dictionary):
    new_sentence = []
    for word in sentences.split():
        new_sentence.append(lexicon_dictionary[word])
    return new_sentence


def encode_data(files_list, encoded_dir, lexicon_dictionary, length_limit=None):
    """
    encodes the dataset into its numeric form given a constructed dictionary

    Parameters:
    ----------
    files_list: list
        the list of files to scan through
    lexicons_dictionary: dict
        the mappings of unique lexicons

    Returns: tuple (dict, int)
        the data in its numeric form, maximum story length
    """
    story_inputs = []
    stories_lengths = []
    #answers_flag = False  # a flag to specify when to put data into outputs list
    limit = length_limit if length_limit is not None else float("inf")

    llprint("Encoding Data ... 0/%d" % (len(files_list)))
    for index, file_path in enumerate(files_list):
        write_path = join(encoded_dir, basename(file_path)[:basename(file_path).rfind('.json')])
        with open(file_path) as data:
            entry = json.load(data)
            for j in range(len(entry)):
                if j == 0:
                    for line in entry[j]:
                        encoded_line = encode_sentences(line, lexicon_dictionary)
                        if len(story_inputs) == 0:
                            story_inputs = encoded_line
                        else:
                            story_inputs.append(lexicon_dictionary["+"])
                            story_inputs.extend(encoded_line)
                    story_inputs.append(lexicon_dictionary["\\"])
                elif j == 1:
                    for i, question_dictionary in enumerate(entry[j]):
                        question = encode_sentences(question_dictionary["question"], lexicon_dictionary)

                        answer = [lexicon_dictionary["="]]
                        answer.extend(encode_sentences(question_dictionary["answer"], lexicon_dictionary))
                        full_list = story_inputs + question + answer
                        with open(write_path + f"{i}.json", 'w+') as write_file:
                            json.dump(full_list, write_file)

                        stories_lengths.append(len(full_list))
        story_inputs = []

        llprint("\rEncoding Data ... %d/%d" % (index + 1, len(files_list)))

    print("\rEncoding Data ... Done!")
    return files_list, stories_lengths


if __name__ == '__main__':
    task_dir = dirname(abspath(__file__))
    options, _ = getopt.getopt(sys.argv[1:], '', ['data_dir=', 'single_train', 'length_limit='])
    data_dir = None
    joint_train = True
    length_limit = None
    training_files = []
    testing_files = []

    if not exists(join(task_dir, 'data')):
        mkdir(join(task_dir, 'data'))
    if not exists(join(task_dir, 'data', 'clean')):
        mkdir(join(task_dir, 'data', 'clean'))

    for opt in options:
        if opt[0] == '--data_dir':
            data_dir = opt[1]
        if opt[0] == '--single_train':
            joint_train = False
        if opt[0] == '--length_limit':
            length_limit = int(opt[1])

    if data_dir is None:
        if exists(join(task_dir, 'data', 'unencoded')):
            data_dir = join(task_dir, 'data', 'unencoded')
        else:
            raise ValueError("data_dir argument cannot be None")

    all_questions = []
    for a_file in listdir(data_dir):
        if a_file.endswith(".json"):
            full_path = join(data_dir, a_file)
            with open(full_path) as json_file:
                a_list = json.load(json_file)
                a_list = clean_data(a_list)
                for i, item in enumerate(a_list):
                    write_path = join(task_dir, 'data', 'clean', f"{item[2].replace(' ', '')}_{a_file}")
                    with open(write_path, 'w+') as write_file:
                        json.dump(item, write_file)
                    if a_file.endswith("train.json"):
                        training_files.append(write_path)
                    else:
                        testing_files.append(write_path)

                all_questions.extend(a_list)

    lexicon_dictionary = create_dictionary(all_questions)
    lexicon_count = len(lexicon_dictionary)

    print(f"There are {len(lexicon_dictionary)} unique words; "
          f"{len([entry for entry in lexicon_dictionary if entry.isnumeric()])} of these words are numbers, "
          f"{len([entry for entry in lexicon_dictionary if entry.isalpha()])} of these words are standard alphabetical words, and "
          f"{len([entry for entry in lexicon_dictionary if entry in punctuation])} of these words are punctuation marks.")
    with open(join(task_dir, 'data', 'dictionary_entries.json'), 'w+') as write_file:
        json.dump(list(lexicon_dictionary.keys()), write_file)


    processed_data_dir = join(task_dir, 'data', 'encoded')
    train_data_dir = join(processed_data_dir, 'train')
    test_data_dir = join(processed_data_dir, 'test')
    if exists(processed_data_dir) and isdir(processed_data_dir):
        rmtree(processed_data_dir)

    mkdir(processed_data_dir)
    mkdir(train_data_dir)
    mkdir(test_data_dir)

    encoded_training_files, training_stories_lengths = encode_data(training_files, train_data_dir, lexicon_dictionary, length_limit)
    encoded_testing_files, testing_stories_lengths = encode_data(testing_files, test_data_dir, lexicon_dictionary, length_limit)

    stories_lengths = np.array(training_stories_lengths + testing_stories_lengths)
    length_limit = np.max(stories_lengths) if length_limit is None else length_limit
    print("Total Number of stories: %d" % (len(stories_lengths)))
    print("Number of stories with lengthens > %d: %d (%% %.2f) [discarded]" % (length_limit,
                                                                               int(np.sum(
                                                                                   stories_lengths > length_limit)),
                                                                               np.mean(stories_lengths >
                                                                                       length_limit) * 100.0))
    print("Number of Remaining Stories: %d" % (len(stories_lengths[stories_lengths <= length_limit])))

    llprint("Saving processed data to disk ... ")

    pickle.dump(lexicon_dictionary, open(join(processed_data_dir, 'lexicon-dict.pkl'), 'wb'))

    joint_train_data = []

    # for filename in encoded_files:
    #     if filename.endswith("test.json"):
    #         pickle.dump(encoded_files[filename], open(join(test_data_dir, basename(filename) + '.pkl'), 'wb'))
    #     elif filename.endswith("train.json"):
    #         if not joint_train:
    #             pickle.dump(encoded_files[filename], open(join(train_data_dir, basename(filename) + '.pkl'), 'wb'))
    #         else:
    #             joint_train_data.extend(encoded_files[filename])
    #
    # if joint_train:
    #     pickle.dump(joint_train_data, open(join(train_data_dir, 'train.pkl'), 'wb'))

    llprint("Done!\n")
