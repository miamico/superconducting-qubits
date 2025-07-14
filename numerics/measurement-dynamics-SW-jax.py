import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental.ode import odeint
import qutip_jax
from qutip import Qobj, tensor, destroy, qeye, basis, sesolve
import numpy as np  # Added for np.cos

# SW functions
def commutator(A, B):
    return A * B - B * A

def compute_generator_S(H0, V):
    energies = H0.diag()
    dim = H0.shape[0]
    V_mat = V.full()
    i, j = jnp.meshgrid(jnp.arange(dim), jnp.arange(dim), indexing='ij')
    delta = energies[i] - energies[j]
    cond = (jnp.abs(delta) > 1e-12) & (i != j)
    S_mat = jnp.where(cond, V_mat[i, j] / delta, 0)
    return Qobj(S_mat, dims=H0.dims, dtype="jax")

def effective_hamiltonian(H, S, order=4):
    H_eff = Qobj(jnp.zeros_like(H.full()), dims=H.dims, dtype="jax")
    current_term = H.copy()
    H_eff += current_term
    fact = 1.0
    for k in range(1, order + 1):
        current_term = commutator(S, current_term)
        fact *= k
        H_eff += current_term / fact
    return H_eff

def transformed_operator(O, S, order=4):
    O_eff = O.copy()
    current_term = O.copy()
    fact = 1.0
    for k in range(1, order + 1):
        current_term = commutator(S, current_term)
        fact *= k
        O_eff += current_term / fact
    return O_eff

# System parameters
omega_c = 5.0
omega_q = 6.0
alpha = -0.3
g = 0.1
N_c = 10
N_q = 5

# Full operators for SW computation
a = tensor(destroy(N_c, dtype="jax"), qeye(N_q, dtype="jax"))
ad = a.dag()
b = tensor(qeye(N_c, dtype="jax"), destroy(N_q, dtype="jax"))
bd = b.dag()
num_c = ad * a
num_q = bd * b

H_c = omega_c * num_c
H_q = omega_q * num_q + (alpha / 2.0) * num_q * (num_q - 1)
H0 = H_c + H_q
V = g * (a * bd + ad * b)
H = H0 + V

# Compute SW
S = compute_generator_S(H0, V)
H_eff = effective_hamiltonian(H, S, order=8)

# Extract effective parameters
diag = H_eff.diag()
E_g = diag[0]
E_01 = diag[1]
E_10 = diag[N_q]
E_11 = diag[N_q + 1]
omega_d_q = float(E_01 - E_g)  # Cast to float
omega_m = float(E_10 - E_g)    # Cast to float
chi = float((E_11 - E_01) - omega_m)  # Cast to float
print(f"SW effective: qubit freq {omega_d_q}, cavity freq (g) {omega_m}, chi {chi}")

# Effective qubit subspace
H_q_eff_mat = jnp.diag(diag[:3] - E_g)
H_q_eff = Qobj(H_q_eff_mat, dims=[[3], [3]], dtype="jax")

# Effective qubit drive operator
b_eff = transformed_operator(b + bd, S, order=4)
b_q_eff_mat = b_eff.full()[:3, :3]
b_q_eff = Qobj(b_q_eff_mat, dims=[[3], [3]], dtype="jax")

# Qubit operators for expectation
bq = destroy(3, dtype="jax")
num_q_q = bq.dag() * bq
num_q2_op = num_q_q * (num_q_q - qeye(3, dtype="jax")) / 2.0

# Initial qubit state
psi0_q = basis(3, 0, dtype="jax")

# Step 2: Calibrate pi pulse
epsilon_q = 0.05
tau_list = jnp.linspace(10, 200, 40)
P1_list = []
P2_list = []
for tau in tau_list:
    def drive_q_func(t, args):
        if 0 <= t <= tau:
            return epsilon_q * np.cos(omega_d_q * t)  # Use np.cos for Python float
        else:
            return 0.0
    H_t = [H_q_eff, [b_q_eff, drive_q_func]]
    tlist_calib = jnp.linspace(0, tau, 40)
    result = sesolve(H_t, psi0_q, tlist_calib, e_ops=[num_q_q, num_q2_op])
    P1_list.append(result.expect[0][-1])
    P2_list.append(result.expect[1][-1])

idx_pi = jnp.argmax(jnp.array(P1_list))
tau_pi = tau_list[idx_pi]
print(f"Pi pulse duration: {tau_pi}, P1: {P1_list[idx_pi]}, P2: {P2_list[idx_pi]}")

# Step 3: Measurement simulation using classical ODE
epsilon_m = 0.02
T_m = 1000.0
kappa = 0.02
tlist_m = jnp.linspace(0, T_m, 1000)
phi = jnp.pi / 2  # unused in code, but present

def alpha_ode(alpha_flat, t, omega_eff, epsilon_m, omega_m, kappa):
    alpha = alpha_flat[0] + 1j * alpha_flat[1]
    dalpha_dt = -1j * omega_eff * alpha - (kappa / 2) * alpha - 1j * epsilon_m * jnp.cos(omega_m * t)
    return jnp.array([jnp.real(dalpha_dt), jnp.imag(dalpha_dt)])

# For |0>
omega_eff_0 = omega_m
sol_0 = odeint(alpha_ode, jnp.array([0.0, 0.0]), tlist_m, omega_eff_0, epsilon_m, omega_m, kappa, atol=1e-8, rtol=1e-6)
alpha_0 = sol_0[:, 0] + 1j * sol_0[:, 1]
exp_a_0 = alpha_0 * jnp.exp(1j * omega_m * tlist_m)
I_0_sw = jnp.real(exp_a_0)
Q_0_sw = jnp.imag(exp_a_0)

# For |1>
omega_eff_1 = omega_m + chi
sol_1 = odeint(alpha_ode, jnp.array([0.0, 0.0]), tlist_m, omega_eff_1, epsilon_m, omega_m, kappa, atol=1e-8, rtol=1e-6)
alpha_1 = sol_1[:, 0] + 1j * sol_1[:, 1]
exp_a_1 = alpha_1 * jnp.exp(1j * omega_m * tlist_m )
I_1_sw = jnp.real(exp_a_1)
Q_1_sw = jnp.imag(exp_a_1)

