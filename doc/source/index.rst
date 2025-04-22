.. SSM documentation master file, created by
   sphinx-quickstart on Wed Apr  2 20:10:39 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Structured State Space Models
=============================

This repository contains the implementation of **Structured State Space Models** as part of the final project for the course *Development Tools for Scientific Computing* held at SISSA during the academic year 2024–2025.

**Authors**: @FilippoOlivo, @GiovanniCanali.

Project Description
-------------------

**State Space Models (SSMs)** are an emerging class of deep learning architectures that have demonstrated significant promise in the domain of sequence modeling. These models have recently established state-of-the-art performance in tasks such as time series forecasting and audio generation, outperforming traditional recurrent and convolutional approaches in both accuracy and efficiency.

This project investigates **structured variants of SSMs** with a focus on computational efficiency and scalability. The following architectures are implemented and evaluated:

- **S4**
- **S6**
- **H3**
- **Gated MLP**
- **Mamba**

The primary objective is to perform a comparative analysis of the aforementioned models based on the following criteria: accuracy, training time, memory consumption.

To ensure consistency and relevance, the evaluation is conducted on synthetic sequence modeling tasks, including
copy task, and selective copy task. These tasks serve as controlled benchmarks to assess the models' ability to retain and manipulate sequential information over long contexts.

Setup Instructions
------------------

Follow these steps to set up the environment:

1. **Clone the repository and navigate into it:**

::

    git clone https://github.com/FilippoOlivo/SSM.git
    cd SSM

2. **Create a Conda environment with Python:**

::

    conda create --name ssm-env python=3.12 -y

3. **Activate the environment:**

::

    conda activate ssm-env

4. **Install the package:**

::

    python -m pip install .

References
----------

- S4 and low-rank S4 blocks: `Efficiently Modeling Long Sequences with Structured State Spaces <https://doi.org/10.48550/arXiv.2111.00396>`_.
- Diagonal S4 block: `On the Parameterization and Initialization of Diagonal State Space Models <https://arxiv.org/pdf/2206.11893>`_.
- H3 model and shift S4 block: `Hungry Hungry Hippos: Towards Language Modeling with State Space Models <https://doi.org/10.48550/arXiv.2212.14052>`_.
- Mamba model and S6 block: `Mamba: Linear-Time Sequence Modeling with Selective State Spaces <https://doi.org/10.48550/arXiv.2312.00752>`_.
- Parallel scan algorithm: `Efficient Parallelization of a Ubiquitous Sequential Computation <https://arxiv.org/abs/2311.06281>`_.
- Swish activation function: `Searching for Activation Functions <https://doi.org/10.48550/arXiv.1710.05941>`_.
   
.. toctree::
   :maxdepth: 4
   :caption: Contents:

   _rst/model
   _rst/trainer
   _rst/dataset
   _rst/cli
   _rst/utils
   


