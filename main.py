from src.circuit import apply_check, encode_psi, decode_outputs
from src.backend import backend, basis_set, coupling
from src.utils import print_state
from src.dnn_predict import dnn_predict
from qiskit import QuantumCircuit, execute, ClassicalRegister
import numpy as np
from tensorflow import keras
import os
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

if not os.path.exists('data/'):
    os.makedirs('data/')

# Load network model
model = keras.models.load_model("dnn_predictor.krs")

# Define theta range
num_theta = 10
thetas = np.linspace(0, np.pi,num_theta,endpoint=False)


# Number of error-correcting block repetition
num_reps = 10

all_data = []
for tt, theta in enumerate(thetas):
    print('Angle:', theta)
    print('Correct state:')
    psi = np.array([(1+np.exp(1j*theta)),  (1-np.exp(1j*theta))])
    psi /= np.sqrt(np.vdot(psi, psi ))
    print_state( psi )
    prob0 = np.real(psi[0]*np.conj(psi[0])) 
    print('Correct_probs :', prob0 )

    syndromes_list = []
    qc = QuantumCircuit(5)
    encode_psi(qc, theta=theta)

    for ii in range(num_reps):
        apply_check(qc, ii)

    creg = ClassicalRegister(4)
    qc.add_register(creg)

    qc.measure(0, creg[0])
    qc.measure(1, creg[1])
    qc.measure(3, creg[2])
    qc.measure(4, creg[3])

    job = execute(qc, backend=backend, basis_gates=basis_set, 
        coupling_map=coupling, shots=1024)
    counts = job.result().get_counts()
    occurrences, syndromes = decode_outputs(counts)

    # Apply ml model on results
    corrected_occurrences = {}
    for meas_state in syndromes:
        for syndrome, n_occ in syndromes[meas_state]:
            if syndrome in ( ('00 '*10)[:-1], ('11 '*10)[:-1] ):
                corrected_state = meas_state
            else:
                error_landscape = dnn_predict(model, [syndrome])[0]
                
                # Check if you have to apply a bitflip and where
                yes_no = error_landscape.sum(axis=1)%2

                corrected_state = np.array([ int(ii) for ii in meas_state ], dtype=int)
                for ii, val in enumerate(yes_no):
                    if val==1:
                        if corrected_state[ii]==0:
                            corrected_state[ii] = 1
                        else:
                            corrected_state[ii] = 0
                    
                corrected_state = ''.join( corrected_state.astype(str) )
            
            if corrected_state in corrected_occurrences:
                corrected_occurrences[corrected_state] += n_occ
            else:
                 corrected_occurrences[corrected_state] = n_occ

    # post-post processing
    true_keys = np.array( [[0]*4, [1]*4])
    true_occ = {'0000':0, '1111': 0}
    for key in corrected_occurrences:
        key_vect = np.array( [int(ii) for ii in key] )
        
        distances = []
        for tk in true_keys:
            distance = np.sum(np.abs( tk- key_vect) )
            distances.append(distance)

        if distances[0] < distances[1]:
            true_occ['0000'] += corrected_occurrences[key]
        else:
            true_occ['1111'] += corrected_occurrences[key]

    # post-post processing
    true_occ_ori = {'0000':0, '1111': 0}
    for key in occurrences:
        key_vect = np.array( [int(ii) for ii in key] )
        
        distances = []
        for tk in true_keys:
            distance = np.sum(np.abs( tk- key_vect) )
            distances.append(distance)

        if distances[0] < distances[1]:
            true_occ_ori['0000'] += occurrences[key]
        else:
            true_occ_ori['1111'] += occurrences[key]



    plot_histogram([occurrences, corrected_occurrences, true_occ_ori, true_occ], figsize=(12, 6),
        legend=['Measured', 'ML-corrected', 'Measured+distance', 'ML-corrected + distance'])
    plt.text(5, 0.57, '$p_{|0\\rangle}$ = '+str(np.round(prob0, 2)), fontsize=16 )

    plt.tight_layout()
    #plt.savefig('images/histo.png')
    plt.show()