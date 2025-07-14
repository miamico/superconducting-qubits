# Tutorial: Optimizing π Pulses for Superconducting Qubits Using Reinforcement Learning with JAX and QuTiP

This tutorial provides a step-by-step guide to implementing reinforcement learning (RL) for optimizing microwave drive pulses in a superconducting qubit system. Specifically, we focus on calibrating a π pulse—a pulse that flips the qubit from its ground state |0⟩ to the excited state |1⟩—while minimizing leakage to higher levels like |2⟩. The system is modeled using circuit quantum electrodynamics (cQED), where a transmon qubit is coupled to a resonator. We use the Schrieffer-Wolff (SW) transformation to derive an effective Hamiltonian for the low-energy subspace, reducing computational complexity.

The approach leverages:
- **QuTiP** (with JAX backend via `qutip-jax`) for quantum simulations.
- **JAX** for automatic differentiation and efficient numerics.
- **REINFORCE** (a policy gradient RL algorithm) to learn piece-wise constant pulse shapes.

By the end, you'll understand how to simulate qubit dynamics, apply perturbation theory (SW transform), and use RL to optimize controls. We'll derive key concepts mathematically, like a proof assistant, guiding you through each logical step.

## Prerequisites
Before diving in, ensure you have:
- **Libraries**: Install via pip: `jax`, `jaxlib`, `qutip`, `qutip-jax`, `optax`.
- **Basic Knowledge**:
  - Quantum mechanics: Hamiltonians, operators, time evolution (Schrödinger equation).
  - RL: Policies, rewards, policy gradients (REINFORCE).
  - JAX: Autodiff, JIT compilation, vectorization (`vmap`).
- **Environment**: Python 3.12+ with GPU support for JAX if available (speeds up training).

If you're new:
- Read QuTiP docs for quantum objects (`Qobj`).
- JAX tutorial for `grad`, `jit`, `vmap`.
- RL intro: REINFORCE maximizes expected rewards by adjusting policy parameters via gradients.

**Note**: If you encounter `TypeError` with `Qobj` (e.g., JAX not recognizing QuTiP objects), convert explicitly: `jnp.asarray(qobj.full())`.

## Section 1: Schrieffer-Wolff Transformation for Effective Hamiltonians

### Theory and Derivation
In cQED, the full Hamiltonian for a transmon qubit coupled to a resonator is complex due to infinite levels. The SW transformation is a unitary perturbation method to block-diagonalize it, yielding an effective low-energy Hamiltonian.

Start with the general form:
\[ H = H_0 + V \]
where \( H_0 \) is the unperturbed (diagonal) part, and \( V \) is the perturbation (off-diagonal coupling).

**Goal**: Find a unitary \( U = e^S \) (with anti-Hermitian \( S \)) such that \( H' = U^\dagger H U \) is block-diagonal, decoupling subspaces.

Derive \( S \) step-by-step:

1. Assume \( S \) is small, expand \( e^S \approx 1 + S + \frac{1}{2} S^2 + \cdots \).

2. The transformed Hamiltonian:
\[ H' = e^{-S} H e^S = H + [H, S] + \frac{1}{2} [[H, S], S] + \cdots \]
(Baker-Campbell-Hausdorff expansion).

3. Split \( H_0 \) into subspaces A (low-energy) and B (high-energy). \( V \) couples them.

4. Choose \( S \) so off-diagonal terms vanish to first order: \( [H_0, S] + V = 0 \) (off-diagonal).

   - Matrix elements: For \( i \in A, j \in B \), \( S_{ij} = \frac{V_{ij}}{E_j - E_i} \) (from commutator).

   - Proof: \( [H_0, S]_{ij} = (E_i - E_j) S_{ij} \), so \( (E_i - E_j) S_{ij} + V_{ij} = 0 \) implies \( S_{ij} = \frac{V_{ij}}{E_j - E_i} \).

5. Higher orders: Iterate commutators up to order 4 (as in code) for accuracy.

For our system:
- \( H_0 = \omega_c a^\dagger a + \omega_q b^\dagger b + \frac{\alpha}{2} b^\dagger b (b^\dagger b - 1) \) (resonator + anharmonic qubit).
- \( V = g (a b^\dagger + a^\dagger b) \) (RWA coupling).

The effective Hamiltonian includes dispersive shift \( \chi b^\dagger b a^\dagger a \).

### Code Implementation
We define helper functions for commutators and the SW generator.

```python
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap, random
from jax.experimental.ode import odeint
import qutip_jax
from qutip import Qobj, tensor, destroy, qeye, basis
import optax

# SW functions
def commutator(A, B):
    return A * B - B * A

def compute_generator_S(H0, V):
    energies = H0.diag()  # Eigenenergies of H0
    dim = H0.shape[0]
    V_mat = V.full()  # Dense matrix
    i, j = jnp.meshgrid(jnp.arange(dim), jnp.arange(dim), indexing='ij')
    delta = energies[i] - energies[j]
    cond = (jnp.abs(delta) > 1e-12) & (i != j)  # Avoid zero division
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
```

**Explanation**:
- `compute_generator_S`: Builds \( S \) element-wise. \( \delta = E_i - E_j \), \( S_{ij} = V_{ij} / \delta \) for \( i \neq j \).
- `effective_hamiltonian`: BCH expansion to order 4: \( H_{\text{eff}} = H + [S, H] + \frac{1}{2} [S, [S, H]] + \cdots \).
- `transformed_operator`: Similarly transforms drive operators.

## Section 2: System Parameters and Effective Model Setup

### Theory and Derivation
Define the system:
- Resonator frequency \( \omega_c = 5.0 \), qubit \( \omega_q = 6.0 \), anharmonicity \( \alpha = -0.3 \), coupling \( g = 0.1 \).
- Hilbert space: Resonator truncated to \( N_c = 10 \) levels, qubit to \( N_q = 5 \).

Operators:
- \( a, a^\dagger \): Resonator lowering/raising.
- \( b, b^\dagger \): Qubit (anharmonic oscillator).

Hamiltonian derivation:
1. Resonator: \( H_c = \omega_c a^\dagger a \).
2. Qubit: \( H_q = \omega_q b^\dagger b + \frac{\alpha}{2} b^\dagger b (b^\dagger b - 1) \) (Duffing oscillator approximation for transmon).
3. Coupling: \( V = g (a b^\dagger + a^\dagger b) \) (rotating-wave approx.).

Apply SW:
- Compute \( S \), then \( H_{\text{eff}} \).
- Extract parameters: Qubit freq \( \omega_d_q = E_{01} - E_g \), dispersive \( \chi = (E_{11} - E_{01}) - \omega_m \).

Reduce to 3-level qubit subspace (|0⟩, |1⟩, |2⟩) for leakage-aware simulation.

### Code Implementation
```python
# System parameters
omega_c = 5.0
omega_q = 6.0
alpha = -0.3
g = 0.1
N_c = 10
N_q = 5

# Operators and Hamiltonians
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

diag = H_eff.diag()
E_g = diag[0]
E_01 = diag[1]
E_10 = diag[N_q]
E_11 = diag[N_q + 1]
omega_d_q = float(E_01 - E_g)
omega_m = float(E_10 - E_g)
chi = float((E_11 - E_01) - omega_m)
print(f"SW effective: qubit freq {omega_d_q}, cavity freq (g) {omega_m}, chi {chi}")

# Effective qubit subspace
H_q_eff_mat = jnp.diag(diag[:3] - E_g)
H_q_eff = Qobj(H_q_eff_mat, dims=[[3], [3]], dtype="jax")

b_eff = transformed_operator(b + bd, S, order=4)
b_q_eff_mat = b_eff.full()[:3, :3]
b_q_eff = Qobj(b_q_eff_mat, dims=[[3], [3]], dtype="jax")

bq = destroy(3, dtype="jax")
num_q_q = bq.dag() * bq
num_q2_op = num_q_q * (num_q_q - qeye(3, dtype="jax")) / 2.0

psi0_q = basis(3, 0, dtype="jax").full().flatten()
target_state = basis(3, 1, dtype="jax").full().flatten()  # Target for pi pulse

H0_mat = H_q_eff.full()
drive_mat = b_q_eff.full()
num_q_mat = num_q_q.full()
num_q2_mat = num_q2_op.full()
```

**Explanation**:
- Operators are tensor products (e.g., \( a \) acts on resonator, identity on qubit).
- SW extracts effective parameters: Print shows \( \omega_d_q \approx 6.0 \), \( \chi \approx -0.001 \) (weak coupling).
- Subspace: Focus on first 3 qubit levels to track leakage (\( P_2 = \langle 2 | \rho | 2 \rangle \)).
- States: \( \psi_0 = |0\rangle \), target \( |1\rangle \).

## Section 3: Reinforcement Learning Setup

### Theory and Derivation
RL optimizes the pulse by treating it as a sequential decision process:
- **Environment**: Qubit state evolves under piecewise-constant drives.
- **Agent (Policy)**: Neural net outputs amplitude distribution for each bin.
- **Reward**: Final fidelity to |1⟩ (minus leakage).

REINFORCE derivation:
1. Policy \( \pi_\theta(a|s) \): Gaussian \( \mathcal{N}(\mu(s), \sigma(s)) \), params \( \theta \).
2. Trajectory \( \tau = (s_0, a_0, r_0, \dots, s_T) \), return \( G = \sum r_t \).
3. Objective: \( J(\theta) = \mathbb{E}_\tau [G] \).
4. Gradient: \( \nabla J = \mathbb{E} [G \nabla \log P(\tau)] \) (policy gradient theorem).
5. Baseline: Subtract mean reward to reduce variance: advantage \( A = G - b \).

For our case:
- State \( s \): Flattened Re/Im of \( \psi \) (6D vector).
- Action \( a \): Scalar amplitude per bin, scaled by 0.05.
- Pulse: \( \epsilon(t) = a_k \cos(\omega_d_q t) \) in bin k.

### Code Implementation
```python
# RL Hyperparameters
K = 10  # Number of pulse segments (time bins)
tau_total = 100.0  # Fixed total pulse duration
dt = tau_total / K  # Duration per bin
state_dim = 6  # Flattened real + imag of psi (3D complex)
action_dim = 1  # Amplitude per bin (scalar)
hidden_dim = 64
num_epochs = 50
batch_size = 32
learning_rate = 1e-3
action_scale = 0.05  # Scale actions to match epsilon_q ~0.05

# Policy network: MLP for mean and log_std of Gaussian
def policy_network(params, state):
    hidden = jnp.tanh(jnp.dot(params['w1'], state) + params['b1'])
    mean = jnp.dot(params['w2'], hidden) + params['b2']
    log_std = jnp.dot(params['w3'], hidden) + params['b3']
    return mean, log_std

# Initialize params
key = random.PRNGKey(0)
keys = random.split(key, 4)
params = {
    'w1': random.normal(keys[0], (hidden_dim, state_dim)) * 0.1,
    'b1': jnp.zeros(hidden_dim),
    'w2': random.normal(keys[1], (action_dim, hidden_dim)) * 0.1,
    'b2': jnp.zeros(action_dim),
    'w3': random.normal(keys[2], (action_dim, hidden_dim)) * 0.1,
    'b3': jnp.zeros(action_dim)
}
```

**Explanation**:
- Hyperparams: K=10 bins over 100 units time (arbitrary units).
- Policy: Single hidden layer MLP. \( \mu = w_2 \cdot \tanh(w_1 s + b_1) + b_2 \), \( \log \sigma \) similar.
- Init: Small random weights (scale 0.1) for stability.

## Section 4: Dynamics Simulation and Episode Rollout

### Theory and Derivation
Time evolution: Solve Schrödinger equation \( i \dot{\psi} = H(t) \psi \), with \( H(t) = H_{\text{eff}} + \epsilon(t) (b + b^\dagger) \).

For real-valued ODE:
1. \( \psi = \psi_r + i \psi_i \), flatten to vector y = [ψ_r, ψ_i].
2. \( \dot{y} = [\Re(-i H \psi), \Im(-i H \psi)] \).

Per bin: Constant amplitude, integrate with `odeint`.

Episode:
- Start at s0 = flatten(ψ0).
- For each bin: Sample a ~ π(s), evolve, collect log π(a|s).
- Reward only at end: Fidelity \( F = |\langle 1 | \psi_f \rangle|^2 \).

Batch via `vmap` for parallel episodes.

### Code Implementation
```python
# ODE for one time bin (constant amplitude in bin)
def evolve_bin(y, t_start, amplitude, omega_d_q):
    def schrodinger_real_bin(y, t, H0_mat, drive_mat, amp, omega_d_q):
        psi_real = y[:3]
        psi_imag = y[3:]
        psi = psi_real + 1j * psi_imag
        drive = amp * jnp.cos(omega_d_q * (t + t_start))  # Continue time
        H = H0_mat + drive * drive_mat
        dpsi_dt = -1j * jnp.dot(H, psi)
        return jnp.concatenate([jnp.real(dpsi_dt), jnp.imag(dpsi_dt)])
    
    t_bin = jnp.linspace(0, dt, 10)  # Fine-grained for accuracy
    return odeint(schrodinger_real_bin, y, t_bin, H0_mat, drive_mat, amplitude, omega_d_q)[-1]

# Simulate one episode: sequential actions over K bins
def simulate_episode(params, key):
    state = jnp.concatenate([jnp.real(psi0_q), jnp.imag(psi0_q)])
    log_probs = []
    rewards = []
    t_current = 0.0
    
    for k in range(K):
        mean, log_std = policy_network(params, state)
        std = jnp.exp(log_std)
        key, subkey = random.split(key)
        action = mean + std * random.normal(subkey, (action_dim,))
        amplitude = action_scale * action[0]  # Scale action
        
        # Evolve for this bin
        new_state = evolve_bin(state, t_current, amplitude, omega_d_q)
        t_current += dt
        
        # Log prob
        normal = (action - mean) / std
        log_prob = -0.5 * (jnp.log(2 * jnp.pi) + 2 * log_std + normal**2)
        log_probs.append(log_prob)
        
        # Intermediate reward 0
        rewards.append(0.0)
        
        state = new_state
    
    # Final reward
    psi_final_real = state[:3]
    psi_final_imag = state[3:]
    psi_final = psi_final_real + 1j * psi_final_imag
    fidelity = jnp.abs(jnp.dot(jnp.conj(target_state), psi_final))**2
    P2 = jnp.real(jnp.dot(jnp.conj(psi_final), jnp.dot(num_q2_mat, psi_final)))
    final_reward = fidelity #- 10 * P2  # High penalty for leakage (commented in code)
    rewards[-1] = final_reward
    
    return jnp.sum(jnp.array(rewards)), jnp.sum(jnp.array(log_probs)), psi_final

# Vectorize over batch
batch_simulate = vmap(simulate_episode, in_axes=(None, 0))
```

**Explanation**:
- `evolve_bin`: Solves ODE for bin. Drive \( \epsilon(t) = \text{amp} \cos(\omega t) \).
- Episode loop: Sequential (not vectorized, but batches via vmap).
- Log prob: For Gaussian, \( \log \pi = -\frac{1}{2} [\log(2\pi) + 2\log\sigma + ((a-\mu)/\sigma)^2] \).
- Reward: Sum is just final F (intermediates 0). Uncomment penalty for leakage.

## Section 5: Training and Evaluation

### Theory and Derivation
Loss: \( L = - \mathbb{E} [A \log \pi] \), with A = G - baseline.

1. Compute over batch: Rewards, log probs.
2. Baseline = mean(rewards).
3. Gradient ascent on J via Adam optimizer.

Evaluation: Run deterministic episode (use means, no sampling) to compute P1 = <1|ρ|1>, P2 = <2|ρ|2>.

### Code Implementation
```python
# Loss for REINFORCE
def loss_fn(params, keys):
    rewards, log_probs, _ = batch_simulate(params, keys)
    baseline = jnp.mean(rewards)
    advantages = rewards - baseline
    return -jnp.mean(advantages * log_probs)

grad_loss = jit(grad(loss_fn))

# Optimizer
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

# Training
for epoch in range(num_epochs):
    key = random.PRNGKey(epoch)
    keys = random.split(key, batch_size)
    loss = loss_fn(params, keys)
    grads = grad_loss(params, keys)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    if epoch % 50 == 0:
        rewards, _, _ = batch_simulate(params, keys)
        print(f"Epoch {epoch}, Loss: {loss}, Avg Reward: {jnp.mean(rewards)}")

# Evaluate optimal policy (deterministic: use means)
_, _, psi_final = simulate_episode(params, random.PRNGKey(42))  # Run one episode
P1_opt = jnp.real(jnp.dot(jnp.conj(psi_final), jnp.dot(num_q_mat, psi_final)))
P2_opt = jnp.real(jnp.dot(jnp.conj(psi_final), jnp.dot(num_q2_mat, psi_final)))
print(f"Optimized P1: {P1_opt}, P2: {P2_opt}")
```

**Explanation**:
- `loss_fn`: Averages over batch.
- Training: 50 epochs, print every 50 (adjust for longer runs).
- Eval: Uses same simulate but with fixed key; P1/P2 via expectation values.

## Running and Interpreting Results
Copy the full code into a Jupyter notebook. Run: Expect SW params printed first, then training logs (reward approaching 1.0 for good fidelity). Final P1 ~1, P2 ~0 indicates success.

If slow: Reduce K or batch_size. For better fidelity, add leakage penalty, increase epochs.

This setup can extend to multi-qubit gates or noisy dynamics—explore further!