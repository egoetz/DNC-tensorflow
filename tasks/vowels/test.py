# -*- coding: utf-8 -*-

from recurrent_controller import RecurrentController
from dnc.dnc import DNC
import tensorflow as tf
import numpy as np
import pickle
import sys
import os
import re


def llprint(message):
    """
    Flushes message to stdout
    :param message: A string to print.
    :return: None.
    """
    sys.stdout.write(message)
    sys.stdout.flush()


def load(path):
    """
    Unpickle the file located at path.
    :param path: The path to the pickled file.
    :return: Returns the object hierarchy stored in the file.
    """
    return pickle.load(open(path, 'rb'))


def onehot(index, size):
    """
    Create a numpy vector that has all zeros except at index. index has the value 1.
    :param index: The index where the vector should be one.
    :param size: The length of the vector.
    :return: A one-hot vector encoding for the given index.
    """
    vec = np.zeros(size, dtype=np.float32)
    vec[index] = 1.0
    return vec


def prepare_sample(sample, word_space_size):
    """
    Transform a sequence of letters and the correct response into an input vector.
    :param sample: list of letters forming word.
    :param word_space_size: how many total letters exist.
    :return: tuple including input vector and the length of its first dimension (i.e. how many one-hot vectors).
    """
    input_vec = np.array(sample[0]['inputs'], dtype=np.float32)
    seq_len = input_vec.shape[0]
    input_vec = np.array([onehot(int(code), word_space_size) for code in input_vec])

    return (
        np.reshape(input_vec, (1, -1, word_space_size)),
        seq_len
    )


def main():
    """
    Tests the latest checkpoint of the DNC that was trained on the vowels task. In this task, the DNC is given an input
    that consist of a sequence of letters and asked to return any vowels contained in that sequence in order of
    their appearance in the sequence. For simplicity's sake, y is not considered a vowel.
    :return: None.
    """
    ckpts_dir = './checkpoints/'
    lexicon_dictionary = load('./data/encoded/lexicon-dict.pkl')
    target_code = lexicon_dictionary["#"]
    test_files = []

    for entry_name in os.listdir('./data/encoded/test/'):
        entry_path = os.path.join('./data/encoded/test/', entry_name)
        if os.path.isfile(entry_path):
            test_files.append(entry_path)

    graph = tf.Graph()
    with graph.as_default():
        with tf.compat.v1.Session(graph=graph) as session:

            ncomputer = DNC(
                RecurrentController,
                input_size=len(lexicon_dictionary),
                output_size=len(lexicon_dictionary),
                max_sequence_length=100,
                memory_words_num=256,
                memory_word_size=64,
                memory_read_heads=4,
            )

            ncomputer.restore(session, ckpts_dir, 'step-100001')

            outputs, _ = ncomputer.get_outputs()
            softmaxed = tf.nn.softmax(outputs)

            tasks_results = {}
            tasks_numbers = []
            counter = 0
            for test_file in test_files:
                test_data = load(test_file)
                task_regexp = r'([0-9]{1,4})test.txt.pkl'
                task_filename = os.path.basename(test_file)
                task_match_obj = re.match(task_regexp, task_filename)
                task_number = task_match_obj.group(1)
                tasks_numbers.append(task_number)
                results = []

                for story in test_data:
                    a_story = np.array(story['inputs'])
                    # Bool vector indicating if the target code is the value at that index in a_story
                    target_mask_1 = (a_story == target_code)
                    target_mask = target_mask_1.copy()
                    # Sets the first target code appearance to False so that it will remain in answer
                    target_mask[np.where(target_mask_1 == True)[0][0]] = False

                    desired_answers = np.array(story['outputs'])
                    input_vec, seq_len = prepare_sample([story], len(lexicon_dictionary))
                    softmax_output = session.run(softmaxed, feed_dict={
                        ncomputer.input_data: input_vec,
                        ncomputer.sequence_length: seq_len
                    })

                    softmax_output = np.squeeze(softmax_output, axis=0)
                    given_answers = np.argmax(softmax_output[target_mask], axis=1)

                    is_correct = True
                    if len(given_answers) != len(desired_answers):
                        is_correct = False
                    else:
                        for i in range(len(given_answers)):
                            if given_answers[i] != desired_answers[i]:
                                is_correct = False
                    if not is_correct:
                        print("\nGiven: ", given_answers)
                        print("Expected: ", desired_answers)
                        results.append(False)
                    else:
                        results.append(True)

                counter += 1
                llprint("\rTests Completed ... %d/%d" % (counter, len(test_files)))

                error_rate = 1. - np.mean(results)
                tasks_results[task_number] = error_rate
            print("\n")
            print("-------------------------------------------------------------------")
            all_tasks_results = [v for _, v in tasks_results.items()]
            results_mean = "%.2f%%" % (np.mean(all_tasks_results) * 100)
            failed_count = "%d" % (np.sum(np.array(all_tasks_results) > 0.05))

            print("%-27s%-27s" % ("Percent Failed", results_mean))
            print("%-27s%-27s" % ("Total Failed", failed_count))


if __name__ == "__main__":
    main()
