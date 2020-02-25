# -*- coding: utf-8 -*-

from recurrent_controller import RecurrentController
import sys
sys.path.append('./dnc')
from dnc.dnc import DNC
import tensorflow as tf
import numpy as np
import pickle
import json
import sys
import os
import re
import csv
import statistics


def llprint(message):
    sys.stdout.write(message)
    sys.stdout.flush()


def load(path):
    return pickle.load(open(path, 'rb'))


def onehot(index, size):
    vec = np.zeros(size, dtype=np.float32)
    vec[index] = 1.0
    return vec


def prepare_sample(sample, target_code, word_space_size):
    input_vec = sample[:sample.index(target_code)]
    output_vec = np.array(sample, dtype=np.float32)
    while len(input_vec) < len(output_vec):
        input_vec.append(target_code)
    input_vec = np.array(input_vec, dtype=np.float32)
    seq_len = input_vec.shape[0]
    weights_vec = np.zeros(seq_len, dtype=np.float32)

    target_mask = (input_vec == target_code)
    weights_vec[target_mask] = 1.0
    input_vec = np.array([onehot(int(code), word_space_size) for code in input_vec])
    output_vec = np.array([onehot(int(code), word_space_size) for code in output_vec])

    return (
        np.reshape(input_vec, (1, -1, word_space_size)),
        np.reshape(output_vec, (1, -1, word_space_size)),
        seq_len,
        np.reshape(weights_vec, (1, -1, 1))
    )


if __name__ == '__main__':
    ckpts_dir = './checkpoints/'
    lexicon_dictionary = load('./data/encoded/lexicon-dict.pkl')
    question_code = lexicon_dictionary["="]
    #target_code = lexicon_dictionary["-"]
    test_files = []

    for entry_name in os.listdir('./data/encoded/test/'):
        entry_path = os.path.join('./data/encoded/test/', entry_name)
        if os.path.isfile(entry_path):
            test_files.append(entry_path)

    for filename in [os.path.join('data', 'annotator1_test.txt'), os.path.join('data', 'annotator2_test.txt')]:
        with open(filename) as csvfile:
            readCSV = csv.reader(csvfile, delimiter='\t')
            question_types = {}
            for row in readCSV:
                task_number = row[0]
                question_type = row[2]
                if task_number in question_types:
                    if len(question_types[task_number]) < 2:
                        question_types[task_number].append(question_type)
                else:
                    question_types[task_number] = [question_type]

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

            test_names = []
            test_data = []
            test_data_types = []
            for counter, test_file in enumerate(test_files):
                task_regexp = r'([0-9])+-([0-9])+_test([0-9])+.json'
                task_filename = os.path.basename(test_file)
                task_match_obj = re.match(task_regexp, task_filename)
                if task_match_obj:
                    with open(test_file) as f:
                        test_data.append(json.load(f))
                    task_number = task_match_obj.group(0)[:task_match_obj.group(0).rfind('_')]
                    if task_number in question_types:
                        task_types = question_types[task_number]
                    else:
                        task_types = None
                    test_data_types.append(task_types)
                    task_name = f"Test {test_file[test_file.find(os.path.basename(test_file)):test_file.rfind('_')]} (Question {test_file[-6]})"
                    test_names.append(task_name)
                    llprint("\r%s ... %d/%d" % (task_name, counter + 1, len(test_files)))

            results = []
            tasks_results = {}
            for i, story in enumerate(test_data):
                question_index = story.index(question_code)

                desired_answers = np.array(story)
                input_vec, _, seq_len, _ = prepare_sample(story, lexicon_dictionary['='], len(lexicon_dictionary))
                softmax_output = session.run(softmaxed, feed_dict={
                    ncomputer.input_data: input_vec,
                    ncomputer.sequence_length: seq_len
                })

                softmax_output = np.squeeze(softmax_output, axis=0)
                given_answers = np.argmax(softmax_output[question_index + 1:], axis=1)

                answers_cursor = 0
                question_grade = []
                targets_cursor = question_index + 1
                while targets_cursor < len(story):
                    question_grade.append(given_answers[answers_cursor] == desired_answers[answers_cursor])
                    answers_cursor += 1
                    targets_cursor += 1
                results.append(np.prod(question_grade))
                if test_data_types[i] is not None:
                    for annotator_str in test_data_types[i]:
                        for char in annotator_str:
                            if char in tasks_results:
                                tasks_results[char].append(np.prod(question_grade))
                            else:
                                tasks_results[char] = [np.prod(question_grade)]

                llprint("\r%s ... %d/%d" % (test_names[i], i, len(test_data)))
            error_rate = 1. - np.mean(results)
            llprint("\r%s ... %.3f%% Error Rate.\n" % (task_name, error_rate * 100))

            print("\n")
            print("%-27s%-27s%s" % ("Task", "Mean", "Standard Deviation"))
            print("-------------------------------------------------------------------")
            means = []
            for task in tasks_results.keys():
                means.append(statistics.mean(tasks_results[task]))
                print("%-27s%-27s%s" % (task, means[-1], statistics.stdev(tasks_results[task])))
            print("-------------------------------------------------------------------")
            results_mean = "%.2f%%" % (np.mean(results) * 100)
            failed_count = "%d" % (np.sum(means < 0.05))

            print("%-27s%-27s" % ("Mean Err.", results_mean))
            failed_count = 0
            for mean in means:
                if mean < .95:
                    failed_count += 1
            print("%-27s%-27s" % ("Failed (err. > 5%)", failed_count))
