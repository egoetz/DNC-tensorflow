import sys
import pickle
import getopt
import numpy as np
from shutil import rmtree
from os import listdir, mkdir
from os.path import join, isfile, isdir, dirname, basename, abspath, exists


def llprint(message):
    """
    Flushes message to stdout
    :param message: A string to print.
    :return: None.
    """
    sys.stdout.write(message)
    sys.stdout.flush()


def create_dictionary(files_list):
    """
    Create a dictionary of unique lexicons in the dataset and their mapping to numbers.
    :param files_list: the list of files to scan through.
    :return: the constructed dictionary of lexicons
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
    print(lexicons_dict)
    return lexicons_dict


def encode_data(files_list, lexicons_dictionary):
    """
    Encode the dataset into its numeric form given a constructed dictionary
    :param files_list: the list of files to scan through.
    :param lexicons_dictionary: the mappings of unique lexicons.
    :return: the data in its numeric form, maximum story length
    """

    files = {}

    llprint("Encoding Data ... 0/%d" % (len(files_list)))

    for indx, filename in enumerate(files_list):

        files[filename] = []

        with open(filename, 'r') as fobj:
            on_answer = False
            story_inputs = []
            story_outputs = []
            for line in fobj:
                word = line.strip()
                if on_answer:
                    story_outputs.append(lexicons_dictionary[word])
                else:
                    story_inputs.append(lexicons_dictionary[word])
                    if word == '#':
                        on_answer = True
            story_inputs.extend([lexicons_dictionary['#']] * len(story_outputs))
            files[filename].append({
                'inputs': story_inputs,
                'outputs': story_outputs
            })

        llprint("\rEncoding Data ... %d/%d" % (indx + 1, len(files_list)))

    print("\rEncoding Data ... Done!")
    return files


def generate_data(directory, total_examples):
    """
    Create a training (9 /10 of total_examples) and testing (1 / 10 of total_examples) files that each contain a
     single example of extracting vowels from a word. Each line in a given text files contains one character. Before
     the '#' character, the lines spell out a word. After the '#' character, the lines repeat the vowels instances in
     that word in their order of occurrence.
    :param directory: The directory in which to store training and testing examples
    :param total_examples: The total number of examples desired
    :return: None.
    """
    word_file = "/usr/share/dict/words"
    words = list(map(str.lower, open(word_file).read().splitlines()))
    for i in range(0, total_examples):
        my_inputs = list(words[i])
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


def main():
    """
    Generate the data used for training the DNC on how to find vowels in words. Create data directories storing this
    information in its unencoded form and encoded form.
    :return: None.
    """
    task_dir = dirname(abspath(__file__))
    options, _ = getopt.getopt(sys.argv[1:], '', ['data_dir=', 'single_train'])
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

    for entry_name in listdir(data_dir):
        entry_path = join(data_dir, entry_name)
        if isfile(entry_path) and entry_path.endswith('txt'):
            files_list.append(entry_path)

    lexicon_dictionary = create_dictionary(files_list)
    lexicon_count = len(lexicon_dictionary)

    encoded_files = encode_data(files_list, lexicon_dictionary)

    print(("Total Number of examples: %d" % (len(encoded_files))))

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


if __name__ == '__main__':
    main()
