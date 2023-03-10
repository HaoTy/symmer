from symmer.operators import PauliwordOp, QuantumState
import numpy as np
import scipy as sp
from typing import Union, List, Tuple
from functools import reduce
import py3Dmol

def exact_gs_energy(
        sparse_matrix, 
        initial_guess=None, 
        n_particles=None, 
        number_operator=None, 
        n_eigs=6
    ) -> Tuple[float, np.array]:
    """ Return the ground state energy and corresponding ground statevector for the input operator
    
    Specifying a particle number will restrict to eigenvectors |ψ> such that <ψ|N_op|ψ> = n_particles
    where N_op is the given number operator.
    """
    # Note the eigenvectors are stored column-wise so need to transpose
    if sparse_matrix.shape[0] > 2**5:
        eigvals, eigvecs = sp.sparse.linalg.eigsh(
            sparse_matrix,k=n_eigs,v0=initial_guess,which='SA',maxiter=1e7
        )
    else:
        # for small matrices the dense representation can be more efficient than sparse!
        eigvals, eigvecs = np.linalg.eigh(sparse_matrix.toarray())
    
    # order the eigenvalues by increasing size
    order = np.argsort(eigvals)
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    
    if n_particles is None:
        # if no particle number is specified then return the smallest eigenvalue
        return eigvals[0], QuantumState.from_array(eigvecs[:,0].reshape([-1,1]))
    else:
        assert(number_operator is not None), 'Must specify the number operator.'
        # otherwise, search through the first n_eig eigenvalues and check the Hamming weight
        # of the the corresponding eigenvector - return the first match with n_particles
        for evl, evc in zip(eigvals, eigvecs.T):
            psi = QuantumState.from_array(evc.reshape([-1,1])).cleanup(zero_threshold=1e-5)
            assert(~np.any(number_operator.X_block)), 'Number operator not diagonal'
            expval_n_particle = 0
            for Z_symp, Z_coeff in zip(number_operator.Z_block, number_operator.coeff_vec):
                sign = (-1) ** np.einsum('ij->i', 
                    np.bitwise_and(
                        Z_symp, psi.state_matrix
                    )
                )
                expval_n_particle += Z_coeff * np.sum(sign * np.square(abs(psi.state_op.coeff_vec)))
            if round(expval_n_particle) == n_particles:
                return evl, QuantumState.from_array(evc.reshape([-1,1]))
        # if a solution is not found within the first n_eig eigenvalues then error
        raise RuntimeError('No eigenvector of the correct particle number was identified - try increasing n_eigs.')


def random_anitcomm_2n_1_PauliwordOp(n_qubits, complex_coeff=True, apply_unitary=True):
    """ Generate a anticommuting PauliOperator of size 2n+1 on n qubits (max possible size)
        with normally distributed coefficients. Generates in structured way then uses Clifford rotation (default)
        to try and make more random (can stop this to allow FAST build, but inherenet structure
         will be present as operator is formed in specific way!)
    """
    base = 'X' * n_qubits
    I_term = 'I' * n_qubits

    P_list = [base]
    for i in range(n_qubits):
        # Z_term
        P_list.append(base[:i] + 'Z' + I_term[i + 1:])
        # Y_term
        P_list.append(base[:i] + 'Y' + I_term[i + 1:])

    coeff_vec = np.random.randn(len(P_list)).astype(complex)
    if complex_coeff:
        coeff_vec += 1j * np.random.randn((len(P_list)))

    P_anticomm = PauliwordOp.from_dictionary((dict(zip(P_list, coeff_vec))))

    # random rotations to get rid of structure
    if apply_unitary:
        U = PauliwordOp.haar_random(n_qubits=n_qubits)
        P_anticomm = U * P_anticomm * U.dagger

    anti_comm_check = P_anticomm.adjacency_matrix.astype(int) - np.eye(P_anticomm.adjacency_matrix.shape[0])
    assert(np.sum(anti_comm_check) == 0), 'operator needs to be made of anti-commuting Pauli operators'

    return P_anticomm

def tensor_list(factor_list:List[PauliwordOp]) -> PauliwordOp:
    """ Given a list of PauliwordOps, recursively tensor from the right
    """
    return reduce(lambda x,y:x.tensor(y), factor_list)


def gram_schmidt_from_quantum_state(state) ->np.array:
    """
    build a unitary to build a quantum state from the zero state (aka state defines first column of unitary)
    uses gram schmidt to find other (orthogonal) columns of matrix

    Args:
        state (np.array): 1D array of quantum state (size 2^N qubits)
    Returns:
        M (np.array): unitary matrix preparing input state from zero state
    """
    state = np.asarray(state).reshape([-1])

    N_qubits = round(np.log2(state.shape[0]))

    missing_amps = 2**N_qubits - state.shape[0]
    state = np.hstack((state, np.zeros(missing_amps, dtype=complex)))

    assert len(state) == 2**N_qubits, 'state is not defined on power of two'
    assert np.isclose(np.linalg.norm(state), 1), 'state is not normalized'

    M = np.eye(2**N_qubits, dtype=complex)

    # reorder if state has 0 amp on zero index
    if np.isclose(state[0], 0):
        max_amp_ind = np.argmax(state)
        M[:, [0, max_amp_ind]] = M[:, [max_amp_ind,0]]

    # defines first column
    M[:, 0] = state
    for a in range(M.shape[0]):
        for b in range(a):
            M[:, a]-= (M[:, b].conj().T @ M[:, a]) * M[:, b]

        # normalize
        M[:, a] = M[:, a] / np.linalg.norm( M[:, a])

    return M

def Draw_molecule(
        xyz_string: str, width: int = 400, height: int = 400, style: str = "sphere"
    ) -> py3Dmol.view:
    """Draw molecule from xyz string.

    Note if molecule has unrealistic bonds, then style should be sphere. Otherwise stick style can be used
    which shows bonds.

    TODO: more styles at http://3dmol.csb.pitt.edu/doc/$3Dmol.GLViewer.html

    Args:
        xyz_string (str): xyz string of molecule
        width (int): width of image
        height (int): Height of image
        style (str): py3Dmol style ('sphere' or 'stick')

    Returns:
        view (py3dmol.view object). Run view.show() method to print molecule.
    """
    view = py3Dmol.view(width=width, height=height)
    view.addModel(xyz_string, "xyz")
    if style == "sphere":
        view.setStyle({'sphere': {"radius": 0.2}})
    elif style == "stick":
        view.setStyle({'stick': {}})
    else:
        raise ValueError(f"unknown py3dmol style: {style}")

    view.zoomTo()
    return view