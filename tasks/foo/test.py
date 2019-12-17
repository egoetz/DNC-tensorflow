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
        tasks_numbers = []
        counter = 0
        for test_file in test_files:
            test_data = load(test_file)
            task_regexp = r'([0-9]{1,3})test.txt.pkl'
            task_filename = os.path.basename(test_file)
            task_match_obj = re.match(task_regexp, task_filename)
            task_number = task_match_obj.group(1)
            tasks_numbers.append(task_number)
            results = []

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
            llprint("\rTests Completed ... %d/%d" % (counter, len(test_files)))

            error_rate = 1. - np.mean(results)
            tasks_results[task_number] = error_rate
        print "\n"
        print "%-27s%-27s" % ("Task", "Result")
        print "-------------------------------------------------------------------"
        for task_number in sorted(tasks_numbers):
            task_result = "%.2f%%" % (tasks_results[task_number] * 100)
            print "%-27s%-27s" % (task_number, task_result)
        print "-------------------------------------------------------------------"
        all_tasks_results = [v for _, v in tasks_results.iteritems()]
        results_mean = "%.2f%%" % (np.mean(all_tasks_results) * 100)
        failed_count = "%d" % (np.sum(np.array(all_tasks_results) > 0.05))

        print "%-27s%-27s" % ("Mean Err.", results_mean)
        print "%-27s%-27s" % ("Failed (err. > 5%)", failed_count)
