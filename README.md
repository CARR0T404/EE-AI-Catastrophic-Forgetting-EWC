# EE-AI-Catastrophic-Forgetting-EWC
# Reproducibility Study: Overcoming Catastrophic Forgetting via Elastic Weight Consolidation (EWC)

This repository contains a comprehensive reproducibility study and implementation of the **Elastic Weight Consolidation (EWC)** algorithm. The project evaluates the effectiveness of EWC in mitigating catastrophic forgetting within sequential learning environments.

## Primary Research Paper
* **Title:** Overcoming catastrophic forgetting in neural networks
* **Authors:** James Kirkpatrick, et al. (DeepMind)
* **Venue:** Proceedings of the National Academy of Sciences (PNAS), 2017

## Theoretical Framework and Secondary Sources
To provide a more rigorous evaluation, this study incorporates frameworks from the following secondary literature:
1. **Measuring Catastrophic Forgetting in Neural Networks** (Kemker et al., 2017): Implementation of standardized metrics to quantify retention and new knowledge acquisition.
2. **Three Scenarios for Continual Learning** (van de Ven & Tolias, 2019): Categorization of experiments into Task-IL, Domain-IL, and Class-IL to identify the operational boundaries of EWC.

---

## Project Overview
Artificial Neural Networks typically suffer from **catastrophic forgetting**: when trained on a new task (Task B), the weights critical for a previous task (Task A) are modified, leading to an abrupt loss of prior expertise.

### The EWC Algorithm
Inspired by synaptic consolidation in biological brains, EWC selectively slows down the learning rate for weights identified as important to previous tasks.
* **Fisher Information Matrix:** The diagonal elements of the Fisher Information matrix are used to estimate the importance of each parameter to a task.
* **Elastic Constraint:** A quadratic penalty is added to the loss function during the training of subsequent tasks. This acts as a "spring" anchoring parameters to their previous optimal values, proportional to their importance.

---

## Experimental Methodology

### 1. Permuted MNIST Protocol
Following the primary paper, we utilize the **Permuted MNIST** task sequence. Each task requires the classification of hand-written digits (0-9), but the input pixels undergo a fixed random permutation for every new task, ensuring distinct input distributions while maintaining the same output space.

### 2. Evaluated Scenarios
We analyze EWC performance across three distinct scenarios defined by van de Ven & Tolias:
* **Task-Incremental Learning (Task-IL):** Task identity is provided during testing.
* **Domain-Incremental Learning (Domain-IL):** Task identity is unknown, but the structure of the task remains consistent.
* **Class-Incremental Learning (Class-IL):** The model must both infer task identity and classify the input, representing the most challenging scenario for EWC.

---

## Quantitative Evaluation Metrics
We adopt the metrics proposed by Kemker et al. to move beyond simple accuracy:
* **Omega Base:** Measures the retention of the model's base knowledge after learning subsequent sessions.
* **Omega New:** Measures the model's ability to learn and immediately recall new tasks.
* **Omega All:** A composite metric evaluating the overall capability of the model to balance retention and new learning.

---

## Implementation Details

## Core Libraries Used
* **Deep Learning:** ```PyTorch``` for model architecture and the EWC penalty loop.  

* **Environment:** ```Gymnasium``` with Atari ROMs for the catastrophic forgetting benchmarks.

* **Computer Vision:** ```OpenCV``` (```cv2```) for image preprocessing and frame warping.

* **Visualization:** ```Matplotlib``` for generating reconstruction and penalty graphs.

* **Data Handling:** ```NumPy``` and ```Pandas``` for matrix operations and result logging.

### Prerequisites
  To run this project, ensure you have Python 3.8+ installed.
  You can install all required dependencies using the provided ```requirements.txt``` file:
  
  ```pip install -r requirements.txt```
