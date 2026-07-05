# Quantum Decoherence Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A C++ tool for **quantifying decoherence** in two-qubit systems subject to local noise channels.  
It identifies **worst-case states** – pure states that are most sensitive to a given noise model – using gradient-based optimization, and validates fundamental properties of composite quantum operations.

## Motivation

In realistic quantum processors, noise drives qubits away from ideal behaviour. Knowing which states are most fragile helps in designing resilient encodings and error mitigation strategies. This project:

- Implements standard single-qubit noise models (depolarizing, amplitude damping, phase damping, etc.) as Kraus operators.
- Constructs a two-qubit composite channel \(\mathcal{E}_1 \otimes \mathcal{E}_2\).
- Quantifies decoherence by the **spectral norm** of the deviation between ideal and noisy density matrices.
- Maximizes this decoherence over all pure two-qubit states via **gradient ascent** with multiple random restarts.
- Verifies the **partial trace consistency**: for any product channel, the reduced dynamics on one qubit equals the local noise applied to the reduced state of that qubit.

This analysis is directly relevant to noise characterization in superconducting qubits, quantum dots, trapped ions, and other solid-state platforms.

## Features

- **Self-contained C++ library** – no external dependencies (all matrix operations are implemented from scratch).
- **Noise models supported:**
  - Depolarizing
  - Amplitude damping
  - Phase damping
  - Coherent (rotation) noise
  - Random (additive) noise
- **Decoherence metric:** spectral norm \(\|\mathcal{E}(\rho) - \rho\|_{\text{sp}}\).
- **Optimization:** gradient ascent with adaptive step size and line search to find the state that maximizes decoherence.
- **Global search:** multiple random initial states to avoid local maxima.
- **Validation suite:** automated test of the partial trace property for composite channels.
- **Visualization-ready output:** saves Bloch vectors of the reduced states of the worst-case state to text files for plotting.

## Code structure
├── main.cpp # Entry point; contains all functions and main()
├── README.md # This file
├── bloch_data_A.txt # Output: Bloch vector of qubit A (generated after run)
└── bloch_data_B.txt # Output: Bloch vector of qubit B (generated after run)

The entire logic is in a single `.cpp` file for clarity, with self-contained classes and functions. In a production setting, one would split it into headers, but this monolithic form makes it easy to compile and experiment with.

## Mathematical details

### Noise modelling
Each single-qubit noise channel \(\mathcal{E}\) is represented by a set of Kraus operators \(\{E_k\}\) such that
\[
\mathcal{E}(\rho) = \sum_k E_k \rho E_k^\dagger.
\]
For two qubits, the composite channel is \(\mathcal{E}_1 \otimes \mathcal{E}_2\), with Kraus operators \(K_{ij} = E_i^{(1)} \otimes F_j^{(2)}\).

### Decoherence measure
For a pure two-qubit state \(|\psi\rangle\) with density matrix \(\rho = |\psi\rangle\langle\psi|\), we define
\[
D(|\psi\rangle) = \left\| \mathcal{E}(\rho) - \rho \right\|_{\text{sp}},
\]
where \(\|\cdot\|_{\text{sp}}\) is the spectral norm (largest singular value). This is a natural distance measure that captures the worst-case deviation over all observables.

### Worst-case state search
We maximize ![D(|\psi\rangle)](https://i.upmath.me/svg/D(%7C%5Cpsi%5Crangle)) over all normalized state vectors \(\psi \in \mathbb{C}^4\).  
The gradient of \(D\) with respect to the real and imaginary parts of the state components is computed via finite differences. Then a **steepest ascent** with line search updates the state:
\[
\psi \leftarrow \text{normalize}\left( \psi + \lambda \nabla D \right),
\]
where the step size \(\lambda\) is chosen to maximize \(D\) among a set of trial values.  
The procedure is repeated for many random starting states to locate the global maximum.

### Partial trace consistency
For any product channel \(\mathcal{E}_1 \otimes \mathcal{E}_2\) and any two-qubit density matrix \(\rho\), the following must hold:
\[
\operatorname{Tr}_2\big[ (\mathcal{E}_1 \otimes \mathcal{E}_2)(\rho) \big] = \mathcal{E}_1\left( \operatorname{Tr}_2[\rho] \right).
\]
The code performs this check numerically for the found worst-case state and for random states, confirming the correctness of the implementation.

## Lindblad dynamics for a physical CNOT gate

Models the time evolution of a controlled‑NOT gate in a double quantum dot charge qubit (Fedichkin & Fedorov, 2003).

- **Hamiltonian**: two‑qubit CNOT with tunable tunnel splitting ε_B.
- **Noise source**: acoustic phonon reservoir.
  - Relaxation of target qubit (σ_−) with rate Γ_B.
  - Dephasing of control qubit (σ_z) with rate Γ_φ.
- **Method**: small‑step Kraus‑operator decomposition (three operators K₀, K₁, K₂ per time step), ensuring complete positivity.
- **Output**: time‑dependent probability P(|1⟩_B), decoherence metric, optimal gate time.

### Requirements
- A C++17 compiler (GCC ≥ 9, Clang ≥ 10, or MSVC 2019+).
- Standard library only – no external packages.

### Compilation
```bash
g++ -std=c++17 -O2 -o decoherence main.cpp

