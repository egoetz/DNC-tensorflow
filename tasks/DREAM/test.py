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


def prepare_sample(sample, target_code, word_space_size, lexicon_dictionary):
    # input_vec = sample[:sample.index(target_code)]
    # output_vec = np.array(sample, dtype=np.float32)
    # while len(input_vec) < len(output_vec):
    #     input_vec.append(target_code)
    # input_vec = np.array(input_vec, dtype=np.float32)
    input_vec = np.array(sample[:sample.index(target_code)], dtype=np.float32)
    output_vec = sample[sample.index(target_code) + 1:]
    while len(output_vec) < len(input_vec):
        output_vec.append(target_code)
    output_vec = np.array(output_vec, dtype=np.float32)
    seq_len = input_vec.shape[0]
    weights_vec = np.zeros(seq_len, dtype=np.float32)

    # target_mask = (input_vec == target_code)
    target_mask = (output_vec != target_code)
    weights_vec[target_mask] = 1.0
    #print("Input vector: ", [list(lexicon_dictionary.keys())[int(num)] for num in input_vec])
    #print("Output vector: ", [list(lexicon_dictionary.keys())[int(num)] for num in output_vec])
    input_vec = np.array([onehot(int(code), word_space_size) for code in input_vec])
    output_vec = np.array([onehot(int(code), word_space_size) for code in output_vec])

    return (
        np.reshape(input_vec, (1, -1, word_space_size)),
        np.reshape(output_vec, (1, -1, word_space_size)),
        seq_len,
        np.reshape(weights_vec, (1, -1, 1))
    )


if __name__ == '__main__':
    ckpts_dir = './checkpoints/word_count_256/'
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

            checkpoints = os.listdir(ckpts_dir)
            if len(checkpoints) != 0:
                checkpoint_numbers = [int(checkpoint[checkpoint.find("-") + 1:]) for checkpoint in checkpoints if
                                      checkpoint[checkpoint.find("-") + 1:].isnumeric()]
                checkpoint_numbers.sort()
                ncomputer.restore(session, ckpts_dir, f"step-{checkpoint_numbers[-1]}")
            else:
                raise FileNotFoundError("No checkpoint to test.")


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
            questions_and_answers = open("test_responses_answer_only.csv", "w+")
            questions_and_answers.write(f"Task Name\tDNC's Answer\tExpected Answer\tGrade\n")
            for i, story in enumerate(test_data):
                question_index = story.index(question_code)

                #desired_answers = np.array(story)
                input_vec, desired_answers, seq_len, _ = prepare_sample(story, lexicon_dictionary['='], len(lexicon_dictionary), lexicon_dictionary)
                desired_answers = [np.where(one_hot_arr == 1)[0][0] for one_hot_arr in desired_answers[0]]
                softmax_output = session.run(softmaxed, feed_dict={
                    ncomputer.input_data: input_vec,
                    ncomputer.sequence_length: seq_len
                })

                softmax_output = np.squeeze(softmax_output, axis=0)
                given_answers = np.argmax(softmax_output, axis=1)

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
                # print("Given Answer: ", given_answers)
                # print("Desired Answer: ", desired_answers)
                # print("Question grade: ", question_grade)
                word_given_answer = [list(lexicon_dictionary.keys())[num] for num in given_answers]
                print(word_given_answer)
                word_desired_answer = [list(lexicon_dictionary.keys())[num] for num in desired_answers]
                questions_and_answers.write(f"{test_names[i]}\t{word_given_answer}\t{word_desired_answer}\t{question_grade}\n")
            questions_and_answers.close()
            error_rate = 1. - np.mean(results)
            llprint("\r%s ... %.3f%% Error Rate.\n" % (task_name, error_rate * 100))

            print("\n")
            print("%-27s%-27s%s" % ("Task", "Mean % Error", "Standard Deviation"))
            print("-------------------------------------------------------------------")
            means = []
            for task in tasks_results.keys():
                means.append(np.mean(tasks_results[task]))
                print("%-27s%-27s%s" % (task, (1 - means[-1]) * 100, statistics.stdev([(1 - result) * 100 for result in tasks_results[task]])))
            print("-------------------------------------------------------------------")
            results_mean = "%.2f%%" % (1 - np.mean(results) * 100)
            failed_count = 0
            for mean in means:
                if mean < .95:
                    failed_count += 1
            failed_count = "%d" % (failed_count)

            print("%-27s%-27s" % ("Mean Err.", results_mean))
            failed_count = 0
            for mean in means:
                if mean < .95:
                    failed_count += 1
            print("%-27s%-27s" % ("Failed (err. > 5%)", failed_count))
