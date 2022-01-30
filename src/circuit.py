from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
import numpy as np

def encode_psi(circ, theta=0, bit_flip=True):
    """Encode the logical state |psi(theta)> on a quantum circuit qc,
    where |psi> = 0.5*( (1+e^{i\theta})|0> + (1-e^{i\theta})|1> ).
    Instead, if looking for phase-flip errors the state is:
    |psi> = 0.707*( |0> + e^{i\theta}|1> )
    

    Parameters
    ----------
    circ : QuantumCircuit
        qiskit quantum circuit
    theta : float
        Angle of rotation along z
    bit_flip : bool, optional
        Flag to choose the type of error to correct.
        If True correct bit-flips, if False phase-flips. Default to True.
    """
    creg = ClassicalRegister(1, 'initialization')
    circ.add_register(creg)
    
    for ii in range(5):
        circ.reset(ii)

    circ.h(2)
    circ.rz(theta, 2)
    circ.h(2)
    if bit_flip:
        for ii in range(5):
            if ii==2:
                continue
            circ.cx(2, ii)

        circ.h(2)
        circ.cz(0, 2)
    else:
        for ii in range(5):
            if ii==2:
                continue
            circ.cz(2, ii)

        circ.h(2)
        circ.cx(0, 2)

    circ.measure(2, creg[0])
    circ.barrier()

def apply_check(circ, check_num, bit_flip=True):
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

def decode_outputs(counts):
    """
    Pass from qiskit standard output to the required format,
    which means two dictionaries that uses as key the final
    measured state:
    - occurrences, with the number of measurements as value
    - syndromes, with a tuple (number of meas, syndrome) as value

    Parameters
    ----------
    counts : dict
        qiskit output dictionary from get_counts

    Returns
    occurrences : dict
        Key is measured final state, value is number of occurrences
    syndromes : dict
        Key is measured final state, value is tuple 
        (number of meas, syndrome)
    """
    occurrences = {}
    error_path = {}
    syndromes = {}
    for key in counts:
        results = key.split(' ')
        
        if results[0] in occurrences:
            occurrences[ results[0]] += counts[key]
            for _ in range(counts[key]):
                error_path[ results[0]].append( ' '.join(results[1:-1][::-1]) )
        else:
            occurrences[ results[0]] = counts[key]
            error_path[ results[0]] = [ ' '.join(results[1:-1][::-1]) ]
            for _ in range(counts[key]-1):
                error_path[ results[0]].append( ' '.join(results[1:-1][::-1]) )

    for key in error_path:
        unique, uni_counts = np.unique(error_path[key], return_counts=True )
        syndromes[key] = [ (unique[ii], uni_counts[ii]) for ii in range(len(unique)) ]

    return occurrences, syndromes