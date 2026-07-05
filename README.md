# Quantum Decoherence Analyzer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A C++ tool for **quantifying decoherence** in two‑qubit systems subject to local noise channels.  
It identifies **worst‑case states** – pure states that are most sensitive to a given noise model – using gradient‑based optimization, and validates fundamental properties of composite quantum operations.  
The project has been extended with a physically motivated **Lindblad simulation of a CNOT gate** in a double quantum dot charge qubit, following Fedichkin & Fedorov (2003).

## Motivation

In realistic quantum processors, noise drives qubits away from ideal behaviour. Knowing which states are most fragile helps in designing resilient encodings and error mitigation strategies. This project:

- Implements standard single‑qubit noise models (depolarizing, amplitude damping, phase damping, etc.) as Kraus operators.
- Constructs a two‑qubit composite channel \(\mathcal{E}_1 \otimes \mathcal{E}_2\).
- Quantifies decoherence by the **spectral norm** of the deviation between ideal and noisy density matrices.
- Maximizes this decoherence over all pure two‑qubit states via **gradient ascent** with multiple random restarts.
- Verifies the **partial trace consistency**: for any product channel, the reduced dynamics on one qubit equals the local noise applied to the reduced state of that qubit.
- Simulates the time evolution of a realistic **CNOT gate** under phonon‑induced relaxation and dephasing, using a small‑step Kraus decomposition of the Lindblad equation.

## Features

- **Self‑contained C++ library** – no external dependencies (all matrix operations are implemented from scratch).
- **Noise models supported:**
  - Depolarizing
  - Amplitude damping
  - Phase damping
  - Coherent (rotation) noise
  - Random (additive) noise
- **Decoherence metric:** spectral norm of the deviation.
- **Optimization:** gradient ascent with adaptive step size and line search to find the state that maximizes decoherence.
- **Global search:** multiple random initial states to avoid local maxima.
- **Validation suite:** automated test of the partial trace property for composite channels.
- **Visualization‑ready output:** saves Bloch vectors of the reduced states of the worst‑case state to text files for plotting.
- **Physical CNOT simulation:** models a charge qubit CNOT gate with phonon noise (relaxation + dephasing) and extracts the optimal gate time.

## Lindblad dynamics for a physical CNOT gate

Models the time evolution of a controlled‑NOT gate in a double quantum dot charge qubit (Fedichkin & Fedorov, 2003).

- **Hamiltonian:** two‑qubit CNOT Hamiltonian with a single tunable parameter ε_B (tunnel splitting).
- **Noise source:** coupling to an acoustic phonon reservoir, described by two Lindblad operators:
  - relaxation of the target qubit (σ_−) with rate Γ_B,
  - pure dephasing of the control qubit (σ_z) with rate Γ_φ.
- **Method:** small‑step Kraus‑operator decomposition of the Lindblad equation. The non‑unitary part is approximated by three Kraus operators K₀, K₁, K₂ per time step, ensuring complete positivity.
- **Output:**
  - time‑dependent probability P(|1⟩_B) of the target qubit,
  - decoherence metric (spectral‑norm deviation from the ideal unitary evolution),
  - optimal gate time that maximizes the success probability before noise dominates.

The implementation is fully integrated with the existing matrix library and shares the same spectral‑norm analysis tools.

## Code structure
├── main.cpp # Entry point; contains all functions and main()
├── README.md # This file
├── bloch_data_A.txt # Output: Bloch vector of qubit A (generated after run)
├── bloch_data_B.txt # Output: Bloch vector of qubit B (generated after run)
└── time_evolution.txt # Output: time‑evolution data from the CNOT simulation

The entire logic is in a single `.cpp` file for clarity. In a production setting one would split it into headers, but this monolithic form makes it easy to compile and experiment with.

## Mathematical details

### Decoherence measure

For a pure two‑qubit state |ψ⟩ with density matrix ρ = |ψ⟩⟨ψ|, the decoherence is defined as

![decoherence metric](https://i.upmath.me/svg/D(%7C%5Cpsi%5Crangle)%20%3D%20%5Cleft%5C%7C%20%5Cmathcal%7BE%7D(%5Crho)%20-%20%5Crho%20%5Cright%5C%7C_%7B%5Cmathrm%7Bsp%7D%7D)

where ‖·‖_sp is the spectral norm (largest singular value). This measure captures the worst‑case deviation over all observables.

### Worst‑case state search

We maximize D over all normalized state vectors ψ ∈ ℂ⁴. The gradient of D with respect to the real and imaginary parts of the state components is computed via finite differences. Then a **steepest ascent** with line search updates the state:

![gradient ascent](https://i.upmath.me/svg/%5Cpsi%20%5Cleftarrow%20%5Ctext%7Bnormalize%7D%5Cleft(%20%5Cpsi%20%2B%20%5Clambda%20%5Cnabla%20D%20%5Cright))

where the step size λ is chosen to maximize D among a set of trial values. The procedure is repeated for many random starting states to locate the global maximum.

### Partial trace consistency

For any product channel \(\mathcal{E}_1 \otimes \mathcal{E}_2\) and any two‑qubit density matrix ρ, the following must hold:

![partial trace consistency](https://i.upmath.me/svg/%5Coperatorname%7BTr%7D_2%5Cbig%5B%20(%5Cmathcal%7BE%7D_1%20%5Cotimes%20%5Cmathcal%7BE%7D_2)(%5Crho)%20%5Cbig%5D%20%3D%20%5Cmathcal%7BE%7D_1%5Cleft(%20%5Coperatorname%7BTr%7D_2%5B%5Crho%5D%20%5Cright))

The code performs this check numerically for the found worst‑case state and for random states, confirming the correctness of the implementation.

## Build and run

### Requirements
- A C++17 compiler (GCC ≥ 9, Clang ≥ 10, or MSVC 2019+).
- Standard library only – no external packages.

### Compilation
```bash
g++ -std=c++17 -O2 -o decoherence main.cpp
