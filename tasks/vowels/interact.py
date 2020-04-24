# -*- coding: utf-8 -*-

from recurrent_controller import RecurrentController
from dnc.dnc import DNC
import tensorflow as tf
import numpy as np
import pickle
import sys
import os


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


def prepare_sample(sample, answers, target_code, word_space_size):
    """
    Transform a sequence of letters and the correct response into input and output vectors.
    :param sample: list of letters forming word.
    :param answers: response that the DNC should give.
    :param target_code: code indicating end of sample and beginning of answer (also used in input as
                        a replacement for each letter in the answer.
    :param word_space_size: how many total letters exist.
    :return: tuple including input vector, output vector, length of sequence, and associated weights.
    """
    input_vec = np.array(sample[0], dtype=np.float32)
    output_vec = np.array(sample[0], dtype=np.float32)
    seq_len = input_vec.shape[0]
    weights_vec = np.zeros(seq_len, dtype=np.float32)

    output_vec = np.append(output_vec, np.array(answers, dtype=np.float32))
    input_vec = np.append(input_vec, np.array([target_code] * len(answers), dtype=np.float32))

    input_vec = np.array([onehot(int(code), word_space_size) for code in input_vec])
    output_vec = np.array([onehot(int(code), word_space_size) for code in output_vec])
    seq_len = input_vec.shape[0]
    return (
        np.reshape(input_vec, (1, -1, word_space_size)),
        np.reshape(output_vec, (1, -1, word_space_size)),
        seq_len,
        np.reshape(weights_vec, (1, -1, 1))
    )


def get_solution(story):
    """
    Find the answer to the question: "What are the instances of vowels (excluding y) contained in story? Repeat them
    in order."
    :param story: the list of letters to find vowels in
    :return: a list of vowels
    """
    story.append('#')
    my_outputs = story + [letter for letter in story if letter in ['a', 'e', 'i', 'o', 'u']]
    return my_outputs


def main():
    """
    Runs an interactive shell where the user can submit input with their chosen deliminator and see the output of the
    DNC's latest checkpoint. 
    :return: None
    """
    dir_path = os.path.dirname(os.path.realpath(__file__))
    ckpts_dir = os.path.join(dir_path, 'checkpoints')
    lexicon_dictionary = load(os.path.join(dir_path, 'data', 'encoded', 'lexicon-dict.pkl'))
    target_code = lexicon_dictionary["#"]

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

            print("This is an interactive shell script. Here a user may test a trained neural network by passing it "
                  "custom inputs and seeing if they elicid the desired output. \n Please note that a user may only "
                  "test inputs that consists of words in the neural network's lexicon. If the user would like to quit"
                  " the program, they can type ':q!' when prompted for an input. \n If the user would like to see the"
                  " network's lexicon, they can type ':dict' when prompted for an input. Otherwise, the user may "
                  "simply type the sequence of inputs that they would like to use and then hit the enter key. \n "
                  "They will then be asked to specify the deliminator that distinguishes one word from another word."
                  " The input will then be split using that deliminator. \n If all resulting inputs are in the "
                  "network's lexicon, the network will then be fed these inputs and its output will be printed for "
                  "the user along with its expected output.")

            my_input = input("Input:")
            while my_input != ":q!":
                if my_input == ":dict":
                    print("The neural network has been trained to recognize the following words:")
                    print(lexicon_dictionary)
                    my_input = input("Input:")
                    continue
                deliminator = input("Deliminator:")
                story = my_input.split(deliminator)
                if not set(story).issubset(lexicon_dictionary):
                    print("You may only test key in the lexicon dictionary.")
                    my_input = input("Input:")
                    continue

                desired_answers = get_solution(story)
                encoded_story = []
                encoded_answers = []
                for an_input in story:
                    encoded_story.append(lexicon_dictionary[an_input])
                for an_output in desired_answers:
                    encoded_answers.append(lexicon_dictionary[an_output])
                input_vec, _, seq_len, _ = prepare_sample([encoded_story], encoded_answers, target_code, 
                                                          len(lexicon_dictionary))
                softmax_output = session.run(softmaxed, feed_dict={
                    ncomputer.input_data: input_vec,
                    ncomputer.sequence_length: seq_len
                })

                softmax_output = np.squeeze(softmax_output, axis=0)
                given_answers = np.argmax(softmax_output[:len(desired_answers)], axis=1)

                print("Output: ", [list(lexicon_dictionary.keys())[list(lexicon_dictionary.values()).index(an_answer)] 
                                   for an_answer in given_answers])
                is_correct = True
                if len(given_answers) != len(encoded_answers):
                    is_correct = False
                else:
                    for i in range(len(given_answers)):
                        if given_answers[i] != encoded_answers[i]:
                            is_correct = False
                if is_correct:
                    print("Correct!")
                else:
                    print("Expected: ", desired_answers)

                my_input = input("Input:")


if __name__ == "__main__":
    main()
