import argparse
import os
import pickle

import numpy as np
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import EfficientSU2
from qiskit.utils import QuantumInstance
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Estimator

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--molecule", type=str, default="Be")
    parser.add_argument("-p", type=int, default=0)
    parser.add_argument("-s", "--seed", type=int, default=42)
    parser.add_argument("-r", "--reps", type=int, default=100)
    parser.add_argument("-t", "--target", type=str, default="H_cs")
    parser.add_argument("--new-vqe", default=False, action="store_true")

    args = parser.parse_args()
    print(args)

    filename = f"{args.molecule}_STO-3G_SINGLET_JW"
    data = pickle.load(open(f"data/cs_op/{filename}.pckl", "rb"))
    H_qiskit = data[args.target].to_qiskit
    n = H_qiskit.num_qubits

    rng = np.random.default_rng(args.seed)

    curr_lowest, best_result, indicator = np.inf, None, "fail"
    for i in range(args.reps):
        ansatz = EfficientSU2(n, reps=args.p)
        optimizer = COBYLA()
        initial_point = rng.uniform(-np.pi, np.pi, ansatz.num_parameters)

        if args.new_vqe:
            from qiskit.algorithms.minimum_eigensolvers import VQE

            vqe = VQE(
                Estimator(
                    backend_options={"method": "statevector"},
                    run_options={"shots": None},
                    approximation=True,
                ),
                ansatz,
                optimizer,
                initial_point=initial_point,
            )
        else:
            from qiskit.algorithms import VQE

            vqe = VQE(
                ansatz,
                optimizer,
                quantum_instance=QuantumInstance(
                    backend=AerSimulator(method="statevector"), shots=None
                ),
            )

        result = vqe.compute_minimum_eigenvalue(H_qiskit)
        if result.optimal_value < data["fci_energy"] + 0.0016:
            indicator = "success"
            break
        if result.optimal_value < curr_lowest:
            curr_lowest = result.optimal_value
            best_result = result

    print(indicator)
    print(best_result)
    dirpath = f"data/vqe/{filename}"
    os.makedirs(dirpath, exist_ok=True)
    pickle.dump(best_result, open(f"{dirpath}/{args.target}-p={args.p}-{indicator}.pckl", "wb"))
