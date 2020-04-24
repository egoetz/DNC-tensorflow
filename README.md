# DNC TensorFlow

This is a TensorFlow implementation of DeepMind's Differentiable Neural Computer (DNC) architecture introduced in their recent Nature paper:
> [Graves, Alex, et al. "Hybrid computing using a neural network with dynamic external memory." Nature 538.7626 (2016): 471-476.](http://www.nature.com/articles/nature20101.epdf?author_access_token=ImTXBI8aWbYxYQ51Plys8NRgN0jAjWel9jnR3ZoTv0MggmpDmwljGswxVdeocYSurJ3hxupzWuRNeGvvXnoO8o4jTJcnAyhGuZzXJ1GEaD-Z7E6X_a9R-xqJ9TfJWBqz)

This implementation includes the bAbI tasks and the copy task from the paper, as implemented by Mostafa-Samir (you can email him at [mostafa.3210@gmail.com](mailto:mostfa.3210@gmail.com) or visit his Github profile [here](https://github.com/Mostafa-Samir)). These tasks have been updated to Python 3 and Tensorflow 2. Additionally, two new tasks have been written and included in the repository: the vowels task and the DREAM task. The vowels task simply trains the DNC to recognize and extract vowels from words (excluding y), while the DREAM task asks the DNC to answer questions on dialogue. Much thanks to Ken Sun et al for publishing [their research and dataset](https://dataset.org/dream/).

## Local Environment Specification

Copy experiments and tests ran on a machine with:
- TensorFlow 2.1
- Python 3.6

bAbI experiment and tests ran on an AWS P2 instance on 1 Tesla K80 GPU as well as a Nvidia GTX 1060 GPU, and on a standard Intel CPU. Note that the difference in running the bAbI and DREAM tasks on a GPU and running the bAbI and DREAM tasks on a CPU is substantial and it is recommended one use a GPU when possible.

## Experiments

### Dynamic Memory Mechanisms

This experiment is designed to demonstrate various functions of the external memory access mechanisms such as in-order retrieval and allocation/deallocation.

A similar approach to that of the paper was followed by training a 2-layer feedforward model with only 10 memory locations on a copy task in which a series of 4 random binary sequences each of which is of size 6 (24 piece of information) was presented as input. Details about the training can be found [here](tasks/copy/).

The model was able to learn to copy the input successfully, and it indeed learned to use the mentioned memory mechanisms. The following figure (which resembles **Extended Data Figure 1** in the paper) illustrates that.

*You can re-generate similar figures in the [visualization notebook](tasks/copy/visualization.ipynb)*

![DNC-Memory-Mechanisms](/assets/DNC-dynamic-mem.png)

- In the **Memory Locations** part of the figure, it's apparent that the model is able to read the memory locations in the same order they were written into.

- In the **Free Gate** and the **Allocation Gate** portions of the figure, it's shown that the free gates are fully activated after a memory location is read and becomes obsolete, while being less activated in the writing phase. The opposite is true for the allocation gate. The **Memory Locations Usage** also demonstrates how memory locations are used, freed, and re-used again time after time.

*The figure differs a little from the one in the paper when it comes to the activation degrees of the gates. This could be due to the small size of the model and the relatively small training time. However, this doesn't affect the operation of the model.*

### Generalization and Memory Scalability

This experiment was designed to check:
- if the trained model has learned an implicit copying algorithm that can be generalized to larger input lengths.
- if the learned model is independent of the training memory size and can be scaled-up with memories of larger sizes.

To approach that, a 2-layer feedforward model with 15 memory locations was trained on a copy problem in which a single sequence of random binary vectors of lengths between 1 and 10 was presented as input. Details of the training process can be found [here](tasks/copy/).

The model was then tested on pairs of increasing sequence lengths and increasing memory sizes with re-training on any of these pairs of parameters, and the fraction of correctly copied sequences out of a batch of 100 was recorded. The model was indeed able to generalize and use the available memory locations effectively without retraining. This is depicted in the following figure which resembles **Extended Data Figure 2** from the paper.

*Similar figures can be re-generated in the [visualization notebook](tasks/copy/visualization.ipynb)*

![DNC-Scalability](/assets/DNC-scalable.png)

### bAbI Task

This experiment was designed to reproduce the paper's results on the bAbI 20QA task. By training a model with the same parameters as DNC1 described in the paper (Extended Data Table 2) on the **en-10k** dataset, the model resulted in error percentages that *mostly* fell within the 1 standard deviation of the means reported in the paper (Extended Data Table 1). The results, and their comparison to the paper's mean results, are shown in the following table. Details about training and reproduction can be found [here](tasks/babi/).

| Task Name | Results | Paper's Mean |
| --------- | ------- | ------------ |
| single supporting fact | 0.00%  | 9.0±12.6% |
| two supporting facts   | 11.88% | 39.2±20.5% |
| three supporting facts | 27.80% | 39.6±16.4% |
| two arg relations      | 1.40%  | 0.4±0.7% |
| three arg relations    | 1.70%  | 1.5±1.0% |
| yes no questions       | 0.50%  | 6.9±7.5% |
| counting               | 4.90%  | 9.8±7.0% |
| lists sets             | 2.10%  | 5.5±5.9% |
| simple negation        | 0.80%  | 7.7±8.3% |
| indefinite knowledge   | 1.70%  | 9.6±11.4% |
| basic coreference      | 0.10%  | 3.3±5.7% |
| conjunction            | 0.00%  | 5.0±6.3% |
| compound coreference   | 0.40%  | 3.1±3.6% |
| time reasoning         | 11.80% | 11.0±7.5% |
| basic deduction        | 45.44% | 27.2±20.1% |
| basic induction        | 56.43% | 53.6±1.9% |
| positional reasoning   | 39.02% | 32.4±8.0% |
| size reasoning         | 8.68%  | 4.2±1.8% |
| path finding           | 98.21% | 64.6±37.4% |
| agents motivations     | 2.71%  | 0.0±0.1% |
| **Mean Err.**          | 15.78% | 16.7±7.6% |
| **Failed (err. > 5%)** |  8     | 11.2±5.4 |

## Vowels
The vowels task was designed to test the DNC's ability to learn a simple task with a procedural solution. Here, the DNC was presented with a sequence of characters forming a word, this word was then followed by a "#" indicating the start of the answer plus a number of additional "#" equal to the number of characters in the answer that the DNC should give. The correct output for the DNC was to repeat the word it was given as well as the first "#", and substitute the remaining "#" symbols with the vowels instances from the original word in their order of appearance. For the sake of simplicity, we exclude 'y'. Trained on 100,000 examples, the DNC was able to pass a test on 1,000 words without getting a single response incorrect.

## DREAM
The DREAM test was designed to push the DNC to the limits of its capabilities and to thwart it. Much of the DNC's difficulty on the bAbI dataset task was found in positional reasoning and basic induction (the two tasks it failed in its original publication). It also had great difficulty when being trained on multiple tasks at once as opposed to being trained and tested on each type of test individually. By presenting the DNC with the DREAM dataset, we were asking it to recognize positions (as each line implicitedly indicated a different speaker), to induce answers (as the DREAM dataset focused on summary and logic questions), and to train on multiple tasks at once (as the DNC does not separate tasks and may actually require multiple skills to be used in a single question). When trained on 100,000 examples in two separate instances, the DNC was unable to solve a single question. A grid search needs to be conducted to confirm these results but preliminary results indicate where the neural network model may be lacking.

## Getting Involved

If you're interested in using the implementation for new tasks, you should first start by **[reading the structure and basic usage guide](docs/basic-usage.md)** to get comfortable with how the project is structured and how it can be extended to new tasks.

If you intend to work with the source code of the implementation itself, you should begin with looking at **[the data flow diagrams](docs/data-flow.md)** to get a high-level overview of how the data moves from the input to the output across the modules of the implementation. This would ease you into reading the source code, which is okay-documented.

You might also find the **[implementation notes](docs/implementation-notes.md)** helpful to clarify how some of the math is implemented.

## License
MIT
