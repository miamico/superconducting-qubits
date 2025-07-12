import numpy as np
from qutip import Qobj, tensor, destroy, qeye, basis, sesolve, expect
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# SW functions
def commutator(A, B):
    """Compute the commutator [A, B] = A B - B A."""
    return A * B - B * A

def compute_generator_S(H0, V):
    """
    Compute the antihermitian generator S for the Schrieffer-Wolff transformation.
    """
    energies = H0.diag()  # Diagonal elements of H0 (unperturbed energies)
    dim = H0.shape[0]
    V_mat = V.full()  # Dense matrix representation of V
    S_mat = np.zeros((dim, dim), dtype=complex)
    
    for i in range(dim):
        for j in range(dim):
            if i != j:
                delta = energies[i] - energies[j]
                if abs(delta) > 1e-12:  # Avoid division by zero for degenerate cases
                    S_mat[i, j] = V_mat[i, j] / delta
    
    return Qobj(S_mat, dims=H0.dims)

def effective_hamiltonian(H, S, order=4):
    """
    Compute the effective Hamiltonian using BCH expansion up to order.
    """
    H_eff = Qobj(np.zeros_like(H.full()), dims=H.dims)
    current_term = H.copy()
    H_eff += current_term
    fact = 1.0
    
    for k in range(1, order + 1):
        current_term = commutator(S, current_term)
        fact *= k
        H_eff += current_term / fact
    
    return H_eff

def transformed_operator(O, S, order=4):
    """
    Compute the transformed operator e^S O e^{-S} ≈ sum (1/k!) [S, ... [S, O]].
    """
    O_eff = O.copy()
    current_term = O.copy()
    fact = 1.0
    
    for k in range(1, order + 1):
        current_term = commutator(S, current_term)
        fact *= k
        O_eff += current_term / fact
    
    return O_eff

# System parameters
omega_c = 5.0  # Cavity frequency
omega_q = 6.0  # Qubit base frequency
alpha = -0.3   # Anharmonicity
g = 0.1        # Coupling strength
N_c = 10        # Cavity levels (unused in reduced sims but for full SW)
N_q = 5        # Qubit levels

# Full operators for SW computation
a = tensor(destroy(N_c), qeye(N_q))
ad = a.dag()
b = tensor(qeye(N_c), destroy(N_q))
bd = b.dag()
num_c = ad * a
num_q = bd * b

H_c = omega_c * num_c
H_q = omega_q * num_q + (alpha / 2.0) * num_q * (num_q - 1)
H0 = H_c + H_q
# V = g * (a + ad) * (b + bd) 
V = g * (a * bd + ad * b)  # Apply RWA to coupling
H = H0 + V

# Compute SW
S = compute_generator_S(H0, V)
H_eff = effective_hamiltonian(H, S, order=8)

# Extract effective parameters from SW (generalized for arbitrary N_q)
diag = H_eff.diag()
E_g = diag[0]  # |0,0>
E_01 = diag[1]  # |0,1>
E_10 = diag[N_q]  # |1,0> = index 1 * N_q + 0
E_11 = diag[N_q + 1]  # |1,1> = index 1 * N_q + 1
omega_d_q = E_01 - E_g
omega_m = E_10 - E_g
chi = (E_11 - E_01) - omega_m
print(f"SW effective: qubit freq {omega_d_q}, cavity freq (g) {omega_m}, chi {chi}")

# Effective qubit subspace (n_c=0 block)
H_q_eff_mat = np.diag([diag[i] - E_g for i in range(3)])
H_q_eff = Qobj(H_q_eff_mat, dims=[[3], [3]])

# Effective qubit drive operator in subspace
b_eff = transformed_operator(b + bd, S, order=4)
b_q_eff_mat = b_eff.full()[0:3, 0:3]
b_q_eff = Qobj(b_q_eff_mat, dims=[[3], [3]])

# Qubit operators for expectation
bq = destroy(3)
num_q_q = bq.dag() * bq
num_q2_op = num_q_q * (num_q_q - qeye(3)) / 2.0

# Initial qubit state
psi0_q = basis(3, 0)

# Step 2: Calibrate pi pulse (reduced qubit sim)
epsilon_q = 0.05
tau_list = np.linspace(10, 200, 40)
P1_list = []
P2_list = []
for tau in tau_list:
    def drive_q_func(t, args):
        if 0 <= t <= tau:
            return epsilon_q * np.cos(omega_d_q * t)
        else:
            return 0.0
    H_t = [H_q_eff, [b_q_eff, drive_q_func]]
    tlist_calib = np.linspace(0, tau, 40)  # Fewer points for speed
    result = sesolve(H_t, psi0_q, tlist_calib, e_ops=[num_q_q, num_q2_op])
    P1_list.append(result.expect[0][-1])
    P2_list.append(result.expect[1][-1])

idx_pi = np.argmax(P1_list)
tau_pi = tau_list[idx_pi]
print(f"Pi pulse duration: {tau_pi}, P1: {P1_list[idx_pi]}, P2: {P2_list[idx_pi]}")

# Step 3: Measurement simulation using classical ODE (decoupled via SW approx)
epsilon_m = 0.02
T_m = 1000.0
kappa = 0.02  # Cavity decay rate (consistent with full sim)
tlist_m = np.linspace(0, T_m, 1000)
phi = np.pi / 2  # Demodulation phase to align I/Q (π/2 swaps roles for |1>)

def alpha_ode(t, alpha_flat, omega_eff, epsilon_m, omega_m, kappa):
    alpha = alpha_flat[0] + 1j * alpha_flat[1]
    dalpha_dt = -1j * omega_eff * alpha - (kappa / 2) * alpha - 1j * epsilon_m * np.cos(omega_m * t)  # Flipped to +1j
    return [np.real(dalpha_dt), np.imag(dalpha_dt)]


# For |0>
omega_eff_0 = omega_m  # Resonant
sol_0 = solve_ivp(alpha_ode, [0, T_m], [0.0, 0.0], args=(omega_eff_0, epsilon_m, omega_m, kappa), t_eval=tlist_m, method='RK45', atol=1e-8, rtol=1e-6)
alpha_0 = sol_0.y[0] + 1j * sol_0.y[1]
exp_a_0 = alpha_0 * np.exp(1j * omega_m * tlist_m)
I_0_sw = np.real(exp_a_0)
Q_0_sw = np.imag(exp_a_0)

# For |1>
omega_eff_1 = omega_m + chi
sol_1 = solve_ivp(alpha_ode, [0, T_m], [0.0, 0.0], args=(omega_eff_1, epsilon_m, omega_m, kappa), t_eval=tlist_m, method='RK45', atol=1e-8, rtol=1e-6)
alpha_1 = sol_1.y[0] + 1j * sol_1.y[1]
exp_a_1 = alpha_1 * np.exp(1j * omega_m * tlist_m )
I_1_sw = np.real(exp_a_1)
Q_1_sw = np.imag(exp_a_1)


# Plot
plt.figure()
plt.plot(I_1_sw[:1000], label='Isw (|1>)', color='red', linestyle = '-.' )
plt.plot(Q_1_sw[:1000], label='Qsw (|1>)', color='blue', linestyle = '-.' )
plt.plot(I_0_sw[:1000], label='Isw (|0>)', color='green', linestyle = '-.' )
plt.plot(Q_0_sw[:1000], label='Qsw (|0>)', color='orange', linestyle = '-.' )
plt.xlabel('Time during measurement')
plt.ylabel('Quadratures')
plt.legend()
plt.show()
