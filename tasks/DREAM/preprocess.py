# -*- coding: utf-8 -*-

import sys
import pickle
import getopt
import json
import numpy as np
import re
from shutil import rmtree
from os import listdir, mkdir
from os.path import join, isdir, dirname, basename, abspath, exists
from string import punctuation
from cleaning import *


def llprint(message):
    """
    Flushes message to stdout
    :param message: A string to print.
    :return: None.
    """
    sys.stdout.write(message)
    sys.stdout.flush()


def clean_sentences(sentence_list):
    """
    Cleans sentence_list by: indicating title words by placing a separate word "\^{}" in front of the capitalized word,
    indicating all-caps word by placing a separate word "\^{}\^{}" in front of the all-caps word, making all words
    lower case, fixing spelling errors, separating units from numbers, giving all onomatopoeia words the same spelling,
    separating two words that were missing a space, changing the ending of possessor nouns ending in s, expanding
    abbreviations, making all number into their own words, changing numbers that were spelled out into digits, making
    all digits into their own separate words and breaking hyphenated words apart.
    :param sentence_list: A list of sentences.
    :return: The modified sentence list.
    """
    new_sentence_list = []
    for sentence in sentence_list:
        if sentence in sentence_dict.keys():
            sentence_list[sentence_list.index(sentence)] = sentence_dict[sentence]

    for index, sentence in enumerate(sentence_list):
        capitalized = set()
        abbreviations = set()
        for word in sentence.split():
            if word.istitle():
                capitalized.add(word)
            elif word.isupper():
                abbreviations.add(word)

        new_sentence = sentence.split()
        for i in range(len(new_sentence)):
            if new_sentence[i] in capitalized:
                new_sentence[i] = f" ^ {new_sentence[i]}"
            if new_sentence[i] in abbreviations:
                new_sentence[i] = f" ^^ {new_sentence[i]}"
        new_sentence = " ".join(new_sentence)
        new_sentence = new_sentence.lower()

        for symbol in symbols.keys():
            if symbol in new_sentence:
                new_sentence = new_sentence.replace(symbol, symbols[symbol])

        words = new_sentence.split()
        for word in words:
            for pattern in spelling_regex.keys():
                match = re.fullmatch(pattern, word)
                if match:
                    new_sentence = new_sentence.split()
                    for i in range(len(new_sentence)):
                        if new_sentence[i] == word:
                            new_sentence[i] = spelling_regex[pattern]
                    new_sentence = " ".join(new_sentence)
            for pattern in units.keys():
                match = re.fullmatch(pattern, word)
                if match:
                    if pattern == r"\d{2,}s-\d{2,}s":
                        new_sentence = new_sentence.split()
                        for i in range(len(new_sentence)):
                            if new_sentence[i] == word:
                                new_sentence[i] = word[:word.find('s')] + " s - " + word[word.find("-") + 1:-1] + " s"
                        new_sentence = " ".join(new_sentence)
                    else:
                        new_sentence = new_sentence.split()
                        for i in range(len(new_sentence)):
                            if new_sentence[i] == word:
                                new_sentence[i] = word[:word.find(units[pattern])] + ' ' + units[pattern]
                        new_sentence = " ".join(new_sentence)
            for pattern in onomatopoeia:
                match = re.fullmatch(pattern, word)
                if match:
                    new_sentence = new_sentence.split()
                    for i in range(len(new_sentence)):
                        if new_sentence[i] == word:
                            new_sentence[i] = onomatopoeia[pattern]
                    new_sentence = " ".join(new_sentence)

        word_array = clean_word_array(new_sentence.split())
        new_sentence = " ".join(word_array)

        new_sentence = new_sentence.replace('-', ' - ')

        serial_names = set()
        for word in new_sentence.split():
            if bool(re.match('^(?=.*[0-9])(?=.*[a-z])', word)):
                serial_names.add(word)
        new_sentence = new_sentence.split()
        for i in range(len(new_sentence)):
            if new_sentence[i] in serial_names:
                new_sentence[i] = ' '.join([f"^ {character}" if character.isalpha() else character for character in
                                            new_sentence[i]])
        new_sentence = " ".join(new_sentence)

        for word in new_sentence.split():
            if word == '×':
                print(new_sentence)

        new_sentence_list.append(new_sentence)

    return new_sentence_list


def clean_data(data):
    """
    Clean strings contained in hierarchical object with the format:
    [
      [
        [
          dialogue 1 / turn 1,
          dialogue 1 / turn 2,
          ...
        ],
        [
          {
            "question": dialogue 1 / question 1,
            "choice": [
              dialogue 1 / question 1 / answer option 1,
              dialogue 1 / question 1 / answer option 2,
              dialogue 1 / question 1 / answer option 3
            ],
            "answer": dialogue 1 / question 1 / correct answer option
          },
          ...
        ],
        dialogue 1 / id
      ],
      ...
    ]
    :param data: hierarchical object formulated as shown above.
    :return: data but with its dialogue, question, and answer cleaned.
    """
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
    Create a dictionary of unique lexicons in the dataset and their mapping to numbers.
    :param data:
    :return:
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
    """
    Change words in sentences into their one-hot index.
    :param sentences: A list of sentences where all words are in lexicon_dictionary
    :param lexicon_dictionary: A dictionary including all the words in the dataset
           sentences are being drawn from.
    :return: sentences with each word replaced by a number.
    """
    new_sentence = []
    for word in sentences.split():
        new_sentence.append(lexicon_dictionary[word])
    return new_sentence


def encode_data(files_list, encoded_dir, lexicon_dictionary):
    """
    Convert open files in files_list, convert their words into numerical equivalents
    as defined in lexicon_dictionary, and then store the encoded information in a
    file of the same name but which is located in encoded_dir.
    :param files_list: The list of files that should have their information converted
    :param encoded_dir: The directory to store the new, encoded files in.
    :param lexicon_dictionary: The dictionary mapping words to numbers (here, numbers
            represent the high bit index for a one-hot vector encoding).
    :return: the list of paths containing the encoded information.
    """
    story_inputs = []
    stories_lengths = []

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
        files_list[index] = write_path

        llprint("\rEncoding Data ... %d/%d" % (index + 1, len(files_list)))

    print("\rEncoding Data ... Done!")
    return files_list, stories_lengths


def main():
    """
    Takes json data files in data_dir in the same format as the DREAM dataset and then creates a new directory
    that contains the same files. But in these files, the dialogue, question and answer's words are cleaned. A second
    new directory is also created, this directory stores the cleaned data in its numerical format.
    their numerical equivalent. If single_train is specified, all training and testing dialogues are dumped into the
    same json file. Otherwise, training and testing data is stored separately.
    :return:
    """
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
    if not exists(join(task_dir, 'data')):
        mkdir(join(task_dir, 'data'))
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

    print(f"There are {len(lexicon_dictionary)} unique words; "
          f"{len([entry for entry in lexicon_dictionary if entry.isnumeric()])} of these words are numbers, "
          f"{len([entry for entry in lexicon_dictionary if entry.isalpha()])} of these words are standard alphabetical"
          f" words, and "
          f"{len([entry for entry in lexicon_dictionary if entry in punctuation])} of these words are punctuation "
          f"marks.")
    print([entry for entry in lexicon_dictionary if not entry.isnumeric() and not entry.isalpha() and entry not in
           punctuation])
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

    encoded_training_files, training_stories_lengths = encode_data(training_files, train_data_dir, lexicon_dictionary)
    encoded_testing_files, testing_stories_lengths = encode_data(testing_files, test_data_dir, lexicon_dictionary)
    encoded_files = encoded_training_files + encoded_testing_files

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

    for filename in encoded_files:
        if filename.endswith("test.json"):
            pickle.dump(encoded_files[filename], open(join(test_data_dir, basename(filename) + '.pkl'), 'wb'))
        elif filename.endswith("train.json"):
            if not joint_train:
                pickle.dump(encoded_files[filename], open(join(train_data_dir, basename(filename) + '.pkl'), 'wb'))
            else:
                joint_train_data.extend(encoded_files[filename])

    if joint_train:
        pickle.dump(joint_train_data, open(join(train_data_dir, 'train.pkl'), 'wb'))

    llprint("Done!\n")


if __name__ == '__main__':
    main()
