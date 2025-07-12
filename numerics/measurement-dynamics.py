import numpy as np
from qutip import Qobj, tensor, destroy, qeye, basis, sesolve, expect, mesolve, Options 

# System parameters (in arbitrary units where frequencies are given)
omega_c = 5.0  # Cavity frequency
omega_q = 6.0  # Qubit base frequency
alpha = -0.3   # Anharmonicity
g = 0.1        # Coupling strength
N_c = 10        # Cavity levels (increased if needed for larger displacements)
N_q = 5        # Qubit levels

# Operators
a = tensor(destroy(N_c), qeye(N_q))
ad = a.dag()
b = tensor(qeye(N_c), destroy(N_q))
bd = b.dag()
num_c = ad * a
num_q = bd * b

# Hamiltonians
H_c = omega_c * num_c
H_q = omega_q * num_q + (alpha / 2.0) * num_q * (num_q - 1)
# V = g * (a + ad) * (b + bd)
V = g * (a * bd + ad * b)  # Apply RWA to coupling
H = H_c + H_q + V

# Initial state |00>
psi0 = tensor(basis(N_c, 0), basis(N_q, 0))

# Step 1: Identify qubit drive frequency using energy levels
## SW effective: qubit freq 6.009900222, cavity freq (g) 17.16774247503247, chi -1.1716477102024179

evals, evecs = H.eigenstates()
# Identify indices by expectations (or sort manually if known)
idx_g = 0  # Ground |0,0>
idx_10 = 1  # Dressed |1,0> (lower energy)
idx_01 = 2  # Dressed |0,1>
idx_11 = 4  # Dressed |1,1> (after |2,0> at idx 3)

E_g = evals[idx_g]
E_10 = evals[idx_10]
E_01 = evals[idx_01]
E_11 = evals[idx_11]

omega_d_q = E_01 - E_g
omega_m = E_10 - E_g
chi = (E_11 - E_01) - omega_m


print(f"effective: qubit freq {omega_d_q}, cavity freq (g) {omega_m}, chi {chi}")

# Step 2: Calibrate pi pulse duration for fixed amplitude
epsilon_q = 0.05  # Drive amplitude (chosen such that Rabi freq << |alpha|)
tau_list = np.linspace(10, 200, 40)  # Scan durations
P1_list = []
P2_list = []
for tau in tau_list:
    def drive_q_func(t, args):
        if 0 <= t <= tau:
            return epsilon_q * np.cos(omega_d_q * t)
        else:
            return 0.0
    H_t = [H, [b + bd, drive_q_func]]
    tlist_calib = np.linspace(0, tau, int(tau / 0.5) + 1)  # Reasonable steps
    result = sesolve(H_t, psi0, tlist_calib, e_ops=[num_q, num_q * (num_q - 1) / 2])
    P1_list.append(result.expect[0][-1])
    P2_list.append(result.expect[1][-1])

# Find tau_pi where P1 max and P2 small
idx_pi = np.argmax(P1_list)
tau_pi = tau_list[idx_pi]
print(f"Pi pulse duration: {tau_pi}, P1: {P1_list[idx_pi]}, P2: {P2_list[idx_pi]}")

# # Step 3: Identify measurement frequency (cavity freq when qubit in |0>)
# idx_c = 1  # Index 1 is ~4.988
# omega_m = evals[idx_c] - E_g
# print(f"Measurement frequency: {omega_m}")

# Step 4: Simulate dynamics with measurement drive for |0> and |1> states
epsilon_m = 0.02  # Measurement drive amplitude (small to avoid high photon number)
T_m = 1000.0  # Measurement duration
kappa = 0.02  # Cavity decay rate (adjust to limit <n> ~ (epsilon_m / kappa)^2)
opts = Options(atol=1e-10, rtol=1e-8)

# For |0>: no qubit drive, measurement from t=0 to T_m
def drive_q_0(t, args):
    return 0.0

def drive_c_0(t, args):
    if 0 <= t <= T_m:
        return epsilon_m * np.cos(omega_m * t)
    else:
        return 0.0

H_t_0 = [H, [b + bd, drive_q_0], [a + ad, drive_c_0]]
tlist_0 = np.linspace(0, T_m, 1000)
result_0 = mesolve(H_t_0, psi0, tlist_0, c_ops=[np.sqrt(kappa) * a], e_ops=[a], options=opts)

# For |1>: qubit pi pulse 0 to tau_pi, measurement tau_pi to tau_pi + T_m
def drive_q_1(t, args):
    if 0 <= t <= tau_pi:
        return epsilon_q * np.cos(omega_d_q * t)
    else:
        return 0.0

def drive_c_1(t, args):
    if tau_pi <= t <= tau_pi + T_m:
        return epsilon_m * np.cos(omega_m * t)
    else:
        return 0.0

H_t_1 = [H, [b + bd, drive_q_1], [a + ad, drive_c_1]]
tlist_1 = np.linspace(0, tau_pi + T_m, int((tau_pi + T_m) / 0.5) + 1)
result_1 = mesolve(H_t_1, psi0, tlist_1, c_ops=[np.sqrt(kappa) * a], e_ops=[a], options=opts)

# Step 5: Compute I, Q quadratures during measurement
# For |0>
exp_a_0 = result_0.expect[0] * np.exp(1j * omega_m * tlist_0)
I_0 = np.real(exp_a_0)
Q_0 = np.imag(exp_a_0)

# For |1>, extract measurement part
idx_start = np.searchsorted(tlist_1, tau_pi)
t_meas_1 = tlist_1[idx_start:] - tau_pi
exp_a_1 = result_1.expect[0][idx_start:] * np.exp(1j * omega_m * (tlist_1[idx_start:] - tau_pi))
I_1 = np.real(exp_a_1)
Q_1 = np.imag(exp_a_1)

# Output or plot (in code, print or use matplotlib for plots)
import matplotlib.pyplot as plt

plt.figure()
plt.plot(t_meas_1, I_1, label='I (|1>)', color='red')
plt.plot(t_meas_1, Q_1, label='Q (|1>)', color='blue')
plt.plot(tlist_0, I_0, label='I (|0>)', color='green')
plt.plot(tlist_0, Q_0, label='Q (|0>)', color='orange')
plt.xlabel('Time during measurement')
plt.ylabel('Quadratures')
plt.legend()
plt.show()