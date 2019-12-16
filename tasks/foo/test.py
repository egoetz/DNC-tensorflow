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
    sys.stdout.write(message)
    sys.stdout.flush()


def load(path):
    return pickle.load(open(path, 'rb'))


def onehot(index, size):
    vec = np.zeros(size, dtype=np.float32)
    vec[index] = 1.0
    return vec


def prepare_sample(sample, target_code, word_space_size):
    input_vec = np.array(sample[0]['inputs'], dtype=np.float32)
    output_vec = np.array(sample[0]['inputs'], dtype=np.float32)
    seq_len = input_vec.shape[0]
    weights_vec = np.zeros(seq_len, dtype=np.float32)

    target_mask = (input_vec == target_code)
    output_vec[target_mask] = sample[0]['outputs']
    weights_vec[target_mask] = 1.0

    input_vec = np.array([onehot(int(code), word_space_size) for code in input_vec])
    output_vec = np.array([onehot(int(code), word_space_size) for code in output_vec])

    return (
        np.reshape(input_vec, (1, -1, word_space_size)),
        np.reshape(output_vec, (1, -1, word_space_size)),
        seq_len,
        np.reshape(weights_vec, (1, -1, 1))
    )


ckpts_dir = './checkpoints/'
lexicon_dictionary = load('./data/encoded/lexicon-dict.pkl')
target_code = lexicon_dictionary["-"]
test_files = []

for entry_name in os.listdir('./data/encoded/test/'):
    entry_path = os.path.join('./data/encoded/test/', entry_name)
    if os.path.isfile(entry_path):
        test_files.append(entry_path)

graph = tf.Graph()
with graph.as_default():
    with tf.Session(graph=graph) as session:

        ncomputer = DNC(
            RecurrentController,
            input_size=len(lexicon_dictionary),
            output_size=len(lexicon_dictionary),
            max_sequence_length=100,
            memory_words_num=256,
            memory_word_size=64,
            memory_read_heads=4,
        )

        ncomputer.restore(session, ckpts_dir, 'step-14976')

        outputs, _ = ncomputer.get_outputs()
        softmaxed = tf.nn.softmax(outputs)

        tasks_results = {}
        tasks_names = {}
        for test_file in test_files:
            test_data = load(test_file)
            task_regexp = r'([0-9]{1,2})test.txt.pkl'
            print task_regexp
            task_filename = os.path.basename(test_file)
            print task_filename
            task_match_obj = re.match(task_regexp, task_filename)
            task_number = task_match_obj.group(1)
            task_name = task_match_obj.group(2).replace('-', ' ')
            tasks_names[task_number] = task_name
            counter = 0
            results = []

            llprint("%s ... %d/%d" % (task_name, counter, len(test_data)))

            for story in test_data:
                a_story = np.array(story['inputs'])
                questions_indices = np.argwhere(a_story == target_code)
                questions_indices = np.reshape(questions_indices, (-1,))
                target_mask = (a_story == target_code)

                desired_answers = np.array(story['outputs'])
                input_vec, _, seq_len, _ = prepare_sample([story], target_code, len(lexicon_dictionary))
                softmax_output = session.run(softmaxed, feed_dict={
                    ncomputer.input_data: input_vec,
                    ncomputer.sequence_length: seq_len
                })

                softmax_output = np.squeeze(softmax_output, axis=0)
                given_answers = np.argmax(softmax_output[target_mask], axis=1)

                answers_cursor = 0
                for question_index in questions_indices:
                    question_grade = []
                    targets_cursor = question_index + 1
                    while targets_cursor < len(a_story) and a_story[targets_cursor] == target_code:
                        question_grade.append(given_answers[answers_cursor] == desired_answers[answers_cursor])
                        answers_cursor += 1
                        targets_cursor += 1
                    results.append(np.prod(question_grade))

                counter += 1
                llprint("\r%s ... %d/%d" % (task_name, counter, len(test_data)))

            error_rate = 1. - np.mean(results)
            tasks_results[task_number] = error_rate
            llprint("\r%s ... %.3f%% Error Rate.\n" % (task_name, error_rate * 100))

        print "\n"
        print "%-27s%-27s%s" % ("Task", "Result", "Paper's Mean")
        print "-------------------------------------------------------------------"
        paper_means = {
            '1': '9.0±12.6%', '2': '39.2±20.5%', '3': '39.6±16.4%',
            '4': '0.4±0.7%', '5': '1.5±1.0%', '6': '6.9±7.5%', '7': '9.8±7.0%',
            '8': '5.5±5.9%', '9': '7.7±8.3%', '10': '9.6±11.4%', '11': '3.3±5.7%',
            '12': '5.0±6.3%', '13': '3.1±3.6%', '14': '11.0±7.5%', '15': '27.2±20.1%',
            '16': '53.6±1.9%', '17': '32.4±8.0%', '18': '4.2±1.8%', '19': '64.6±37.4%',
            '20': '0.0±0.1%', 'mean': '16.7±7.6%', 'fail': '11.2±5.4'
        }
        for k in range(20):
            task_id = str(k + 1)
            task_result = "%.2f%%" % (tasks_results[task_id] * 100)
            print "%-27s%-27s%s" % (tasks_names[task_id], task_result, paper_means[task_id])
        print "-------------------------------------------------------------------"
        all_tasks_results = [v for _, v in tasks_results.iteritems()]
        results_mean = "%.2f%%" % (np.mean(all_tasks_results) * 100)
        failed_count = "%d" % (np.sum(np.array(all_tasks_results) > 0.05))

        print "%-27s%-27s%s" % ("Mean Err.", results_mean, paper_means['mean'])
        print "%-27s%-27s%s" % ("Failed (err. > 5%)", failed_count, paper_means['fail'])
