# PROJECT_PYPY
פרויקט ובו אנו נשחזר ניסוי במאמר# Overcoming Catastrophic Forgetting in Neural Networks - Reproducibility Project

This repository contains a reproducibility study of the paper:
**"Overcoming catastrophic forgetting in neural networks"** by Kirkpatrick et al. (DeepMind, 2016).

## Project Overview
The goal of this project is to recreate the scientific results demonstrating **Elastic Weight Consolidation (EWC)**. This algorithm allows neural networks to learn tasks sequentially without forgetting previously learned information—a phenomenon known as "Catastrophic Forgetting."

## Methodology
In this project, we focus on the **Permuted MNIST** experiment:
1. **Task A:** Training a multi-layer perceptron (MLP) on the standard MNIST dataset.
2. **Task B:** Training the same network on a version of MNIST where pixels are randomly (but consistently) permuted.
3. **Comparison:** We compare a standard training approach (SGD) against the **EWC** algorithm to observe how EWC preserves the accuracy of Task A while learning Task B.

## Key Features
* Implementation of the **Fisher Information Matrix** to estimate weight importance.
* Comparative visualization of accuracy decay vs. consolidation.
* Python-based implementation using modern deep learning libraries.

## Repository Structure
* `main.py`: The core script for running the experiments.
* `EWC_Implementation.ipynb`: Detailed walkthrough of the code and logic.
* `takeaways.pdf`: A reflective report on the findings and AI-assisted workflow.
* `AI_Work_Log.md`: Documentation of the interaction with AI tools throughout the project.

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run the analysis: `python main.py`.

## References
Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proceedings of the National Academy of Sciences*, 114(13), 3521-3526. 
