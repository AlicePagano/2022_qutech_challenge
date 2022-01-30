from turtle import back
from qiskit.providers.aer import QasmSimulator
import qiskit.providers.aer.noise as noise
from qiskit.providers.aer.noise.errors.standard_errors import thermal_relaxation_error, reset_error
import numpy as np


# Basis set of gates
basis_set = ['x', 'y', 'id', 'rx', 'ry', 'rz', 'cz']

# Coupling map
coupling = [ [0, 2], [1, 2], [3, 2], [4, 2], 
                [2, 0], [2, 1], [2, 3], [2, 4] ]

# Relaxation time for the qubits
t1s = np.array([12.5, 12.9, 20.4, 13.6, 13.4])*1e-6
# Dephasing time for the qubits
t2s = np.array([22.1, 22.6, 21.7, 18.3, 19.8])*1e-6
# Fidelity of single-qubit operations
f1q = np.array([0.999, 0.998, 0.998, 0.996, 0.999])
# Fidelity of two-qubit operations
f2q = np.array([0.976, 0.977, None, 0.971, 0.975])
# Initialization fidelity
finit = np.array([0.996, 0.996, 0.99, 0.972, 0.908])
# Readout fidelity
freadout = np.array([0.923, 0.945, 0.968, 0.934, 0.96])

# Time of 2-qubit gates
t_2q = 60*1e-9
# Time of single-qubit gates
t_1q = 20*1e-9

noise_model = noise.NoiseModel()

# Depolarizing probabilities (to check with stardom for real prob)
depolarizing_prob = 0.001
depolarizing_prob2 = 0.002

#noise_model.add_all_qubit_quantum_error(
#    noise.depolarizing_error(depolarizing_prob, 1),
#    ['x', 'y', 'id', 'rx', 'ry', 'rz']
#)
noise_model.add_all_qubit_quantum_error(
    noise.depolarizing_error(depolarizing_prob2, 2),
    ['cz']
)


for ii in range(5):
    # Add phase and amplitude damping channel for single-qubit gates
    noise_model.add_quantum_error( 
        thermal_relaxation_error(t1s[ii], t2s[ii], t_1q).compose(noise.depolarizing_error(depolarizing_prob, 1)) ,
        ['x', 'y', 'id', 'rx', 'ry', 'rz'],
        [ii]
    )
    
    # Add phase and amplitude damping channel for two-qubits gate
    #noise_model.add_quantum_error( 
    #    thermal_relaxation_error(t1s[ii], t2s[ii], t_2q),
    #    ['cz'],
    #    [ii]
    #)

    # Add readout error
    #noise_model.add_readout_error(
    #    noise.ReadoutError(1- freadout[ii]), 
    #    ii
    #)

    # Add initialization error by adding reset error and starting
    # simulation with a reset
    noise_model.add_quantum_error( reset_error(finit[ii]), 
        'reset',
        [ii])



backend = QasmSimulator(method='density_matrix',
                        noise_model = noise_model)
