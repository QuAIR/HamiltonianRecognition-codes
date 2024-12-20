# HamiltonianRecognition-codes Documentation

Source code for the paper [*Optimal Hamiltonian recognition of unknown quantum dynamics*](https://arxiv.org/abs/2412.13067).

Identifying unknown Hamiltonians from their quantum dynamics is a pivotal challenge in quantum technologies and fundamental physics. Based on two quantum signal processing (QSP) structures, one can develop quantum algorithms to differentiate rotations with unknown angles, governed by unknown qubit-Hamiltonian selected from a known set.

In this repository, we provide two algorithm examples: one for binary Hamiltonian recognition that differentiate X, Z rotations, and the other for ternary Hamiltonian recognition that differentiate X, Y, Z rotations.

| Protocols in this paper      | Location in this repository                            |
|--------------------|------------------------------------------------------------------|
| [binary Hamiltonian recognition (real experiment)](./qpu_experiment/XZ_recognition.ipynb)   | `qpu_experiment/XZ_recognition.ipynb`|
| [binary Hamiltonian recognition (local simulation)](./verification/XZ_recognition.ipynb)    | `verification/XZ_recognition.ipynb`|
| [ternary Hamiltonian recognition (local simulation)](./verification/XYZ_recognition.ipynb)  | `verification/XYZ_recognition.ipynb`|

## How to Run These Files?

We recommend running these files by creating a virtual environment using `conda` and install Jupyter Notebook. We recommend using Python `3.10` for compatibility.

```bash
conda create -n hamiltonian python=3.10
conda activate hamiltonian
conda install jupyter notebook
```

QPU experiments are based on the cloud platform provided by [Tencent Quantum Lab](https://github.com/tencent-quantum-lab). For readers who are interest in QPU experiments, please use the following command to install the necessary package:

```bash
pip install "tensorcircuit[cloud]"
```

Local simulations are based on the [QuAIRKit](https://github.com/QuAIR/QuAIRKit) package no lower than v0.3.0. This package is featured for batch and qudit computations in quantum information science and quantum machine learning. For readers who are interest in local simulations, run the following commands:

```bash
pip install quairkit
```

## System and Package Versions

It requires tokens to run the QPU experiments. Other than that, no special hardware requirements are needed. The following are the versions of the packages used in this repository:

- tensorcircuit: 0.12.0
- quairkit: 0.3.0
- torch: 2.5.1+cpu
- numpy: 1.26.0
- scipy: 1.14.1
