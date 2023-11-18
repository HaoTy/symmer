import argparse
import json
import pickle
from re import A

import numpy as np

from symmer.operators import PauliwordOp, QuantumState
from symmer.projection import ContextualSubspace, QubitTapering
from symmer.utils import exact_gs_energy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--molecule", type=str, default="Be")
    args = parser.parse_args()
    print(args)

    filename = f"{args.molecule}_STO-3G_SINGLET_JW"
    with open(f"tests/hamiltonian_data/{filename}.json", "r") as infile:
        data_dict = json.load(infile)
    print(data_dict["data"])

    fci_energy = data_dict["data"]["calculated_properties"]["FCI"]["energy"]
    hf_state = QuantumState(
        np.asarray(data_dict["data"]["hf_array"])
    )  # Hartree-Fock state
    hf_energy = data_dict["data"]["calculated_properties"]["HF"]["energy"]
    H = PauliwordOp.from_dictionary(data_dict["hamiltonian"])

    QT = QubitTapering(H)
    print(
        f"Qubit tapering permits a reduction of {H.n_qubits} -> {H.n_qubits-QT.n_taper} qubits.\n"
    )
    UCC_q = PauliwordOp.from_dictionary(
        data_dict["data"]["auxiliary_operators"]["UCCSD_operator"]
    )

    H_taper = QT.taper_it(ref_state=hf_state)
    UCC_taper = QT.taper_it(aux_operator=UCC_q)

    cs_vqe = ContextualSubspace(
        H_taper,
        noncontextual_strategy="SingleSweep_magnitude",
        unitary_partitioning_method="LCU",
    )

    for n in range(1, H.n_qubits - QT.n_taper + 1):
        try:
            cs_vqe.update_stabilizers(
                n_qubits=n, strategy="aux_preserving", aux_operator=UCC_taper
            )
            # hf_cs = cs_vqe.project_state_onto_subspace(QT.tapered_ref_state)
            H_cs = cs_vqe.project_onto_subspace()
            print(H_cs)
            gs_nrg, gs_psi = exact_gs_energy(H_cs.to_sparse_matrix)
            if gs_nrg <= fci_energy + 0.0016:
                break
        except Exception as e:
            print(n, e)
            continue

    pickle.dump(
        {
            "H": H.to_dictionary,
            "H_taper": H_taper.to_dictionary,
            "H_cs": H_cs.to_dictionary,
            "cs_state": gs_psi,
            "n_qubits": H.n_qubits,
            "n_taper": H.n_qubits - QT.n_taper,
            "n_cs": n,
            "fci_energy": fci_energy,
        },
        open(f"data/cs_op/{filename}.pckl", "wb"),
    )
