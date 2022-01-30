from circuit import apply_check, encode_psi, decode_outputs
from backend import backend, basis_set, coupling
from qiskit import QuantumCircuit, execute, ClassicalRegister
from qiskit.providers.aer import QasmSimulator
import numpy as np
import os
from tqdm import tqdm


def apply_check_with_error(circ, check_num, bit_flip=True):
    """Apply a check on the quantum circuit

    Parameters
    ----------
    circ : qc
        qiskit quantum circuit
    check_num : int
        Number of the check
    bit_flip : bool, optional
        Flag to choose the type of error to correct.
        If True correct bit-flips, if False phase-flips. Default to True.
    """
    if not isinstance(circ, QuantumCircuit):
        raise TypeError()

    error_prob = 1e-2
    stack = np.zeros( (4, 1 ))
    for ii in [0, 1, 3, 4]:
        if np.random.rand() < error_prob:
            circ.x(ii)
            if ii >1:
                ii -= 1
            #print('ERROR')
            stack[ii, 0] = 1

    creg = ClassicalRegister(2, f'check{check_num}')

    permutations = np.array([0, 1, 3, 4])
    idx = check_num%4
    ctrls = np.roll(permutations, idx)

    circ.add_register(creg)
    
    if bit_flip:
        circ.cx(ctrls[0], 2)
        circ.cx(ctrls[1], 2)
    else:
        circ.cz(ctrls[0], 2)
        circ.cz(ctrls[1], 2)
    circ.measure(2, creg[0])
    
    if bit_flip:
        circ.cx(ctrls[1], 2)
        circ.cx(ctrls[2], 2)
    else:
        circ.h(2)
        circ.cz(ctrls[1], 2)
        circ.cz(ctrls[2], 2)
        circ.h(2)
    circ.measure(2, creg[1])

    return stack


if not os.path.exists('data/'):
    os.makedirs('data/')

# Define theta range
num_theta = 10
theta = np.linspace(0, np.pi,num_theta,endpoint=False)

# Number of algorithm execution for each theta
num_times = 1

# Number of error-correcting block repetition
num_reps = 10

features = []
labels = []

for ii in tqdm(range(100000)):
    error_map = np.zeros((4, 1))

    qc = QuantumCircuit(5)
    encode_psi(qc, theta=np.pi/2)
    for ii in range(num_reps):
        new_layer = apply_check_with_error(qc, ii)
        error_map = np.hstack( (error_map, new_layer))
    error_map = error_map[:, 1:]
    labels.append(error_map)
    creg = ClassicalRegister(4)
    qc.add_register(creg)

    qc.measure(0, creg[0])
    qc.measure(1, creg[1])
    qc.measure(3, creg[2])
    qc.measure(4, creg[3])

    job = execute(qc, backend=QasmSimulator(), basis_gates=basis_set, coupling_map=coupling, shots=1)
    counts = job.result().get_counts()
    occurrences, syndromes = decode_outputs(counts)
    features.append( list(list(syndromes.values())[0])[0] )
    #print(list(list(syndromes.values())[0])[0])


np.save('data/features.npy', features )
np.save('data/labels.npy', labels )