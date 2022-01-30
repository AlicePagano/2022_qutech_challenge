import numpy as np
from qiskit import execute, Aer


def print_state(dense_state):
    """
    Prints a *dense_state* with kets. Compatible with quimb states.

    Parameters
    ----------
    dense_state: array_like
            Dense representation of a quantum state

    Returns
    -------
    None: None
    """

    NN = int(np.log2(len(dense_state)))

    binaries = [bin(ii)[2:] for ii in range(2**NN)]
    binaries = ['0'*(NN-len(a)) + a for a in binaries] #Pad with 0s

    ket = []
    for ii, coef in enumerate(dense_state):
        if not np.isclose(np.abs(coef), 0.):
            if np.isclose(np.imag(coef), 0.):
                if np.isclose(np.real(coef), 1.):
                    ket.append('|{}>'.format(binaries[ii]))
                else:
                    ket.append('{:.3f}|{}>'.format(np.real(coef), binaries[ii]))
            else:
                ket.append('{:.3f}|{}>'.format(coef, binaries[ii]))
    print(' + '.join(ket))

def qiskit_get_statevect(qc, backend = Aer.get_backend('statevector_simulator')):
    """
        Returns the statevector of the qiskit quantum circuit *qc*

        Parameters
        ----------
        qc: Quantum circuit
            Quantum circuit of which we want the statevector

        Returns
        -------
        st: array_like
            Statevector of the quantum circuit after the application
            of the reverse operation on the qubit's ordering
    """
    statevector = execute(qc, backend).result()
    statevector = statevector.get_statevector()

    return statevector