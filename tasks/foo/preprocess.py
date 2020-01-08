import sys
import pickle
import getopt
import numpy as np
from shutil import rmtree
from os import listdir, mkdir
from os.path import join, isfile, isdir, dirname, basename, normpath, abspath, exists
import random

def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()

def create_dictionary(files_list):
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

    llprint("Creating Dictionary ... 0/%d" % (len(files_list)))
    for indx, filename in enumerate(files_list):
        with open(filename, 'r') as fobj:
            for line in fobj:
                word = line.strip()
                if not word.lower() in lexicons_dict:
                    lexicons_dict[word.lower()] = id_counter
                    id_counter += 1

        llprint("\rCreating Dictionary ... %d/%d" % ((indx + 1), len(files_list)))

    print("\rCreating Dictionary ... Done!")
    print lexicons_dict
    return lexicons_dict


def encode_data(files_list, lexicons_dictionary):
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

    files = {}

    llprint("Encoding Data ... 0/%d" % (len(files_list)))

    for indx, filename in enumerate(files_list):

        files[filename] = []

        with open(filename, 'r') as fobj:
            onAnswer = False
            story_inputs = []
            story_outputs = []
            for line in fobj:
                word = line.strip()
                if onAnswer:
                    story_outputs.append(lexicons_dictionary[word])
                else:
                    story_inputs.append(lexicons_dictionary[word])
                    if word == '#':
                        onAnswer = True
            story_inputs.extend(lexicons_dictionary['#'] for word in list(range(len(story_outputs))))
            files[filename].append({
                'inputs': story_inputs,
                'outputs': story_outputs
            })

        llprint("\rEncoding Data ... %d/%d" % (indx + 1, len(files_list)))

    print("\rEncoding Data ... Done!")
    return files


def generate_data(directory, total_examples):
    word_file = "/usr/share/dict/words"
    WORDS = map(str.lower, open(word_file).read().splitlines())
    for i in range(0, total_examples):
        my_inputs = list(WORDS[i])
        if i < np.floor(total_examples * 9 / 10):
            path = join(directory, "%dtrain.txt" %i)
        else:
            path = join(directory, "%dtest.txt" %i)
        with open(path, "w+") as file:
            my_outputs = [letter for letter in my_inputs if letter in ['a', 'e', 'i', 'o', 'u']]
            for letter in my_inputs:
                file.write("%c\n" % letter)
            file.write("#\n")
            for letter in my_outputs:
                file.write("%c\n" % letter)


if __name__ == '__main__':
    task_dir = dirname(abspath(__file__))
    options,_ = getopt.getopt(sys.argv[1:], '', ['data_dir=', 'single_train'])
    joint_train = True
    files_list = []
    total_examples = 10000

    if not exists(join(task_dir, 'data')):
        mkdir(join(task_dir, 'data'))
    if not exists(join(task_dir, 'data', 'unencoded')):
        mkdir(join(task_dir, 'data', 'unencoded'))
    generate_data(join(task_dir, 'data', 'unencoded'), total_examples)
    data_dir = join(task_dir, 'data', 'unencoded')

    for opt in options:
        if opt[0] == '--single_train':
            joint_train = False

    for entryname in listdir(data_dir):
        entry_path = join(data_dir, entryname)
        if isfile(entry_path) and entry_path.endswith('txt'):
            files_list.append(entry_path)

    lexicon_dictionary = create_dictionary(files_list)
    lexicon_count = len(lexicon_dictionary)

    encoded_files = encode_data(files_list, lexicon_dictionary)

    print("Total Number of examples: %d" % (len(encoded_files)))

    processed_data_dir = join(task_dir, 'data', 'encoded')
    train_data_dir = join(processed_data_dir, 'train')
    test_data_dir = join(processed_data_dir, 'test')
    if exists(processed_data_dir) and isdir(processed_data_dir):
        rmtree(processed_data_dir)

    mkdir(processed_data_dir)
    mkdir(train_data_dir)
    mkdir(test_data_dir)

    llprint("Saving processed data to disk ... ")

    pickle.dump(lexicon_dictionary, open(join(processed_data_dir, 'lexicon-dict.pkl'), 'wb'))

    joint_train_data = []

    for filename in encoded_files:
        if filename.endswith("test.txt"):
            pickle.dump(encoded_files[filename], open(join(test_data_dir, basename(filename) + '.pkl'), 'wb'))
        elif filename.endswith("train.txt"):
            if not joint_train:
                pickle.dump(encoded_files[filename], open(join(train_data_dir, basename(filename) + '.pkl'), 'wb'))
            else:
                joint_train_data.extend(encoded_files[filename])

    if joint_train:

        pickle.dump(joint_train_data, open(join(train_data_dir, 'train.pkl'), 'wb'))

    llprint("Done!\n")