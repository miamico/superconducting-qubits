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

psi0_q = basis(3, 0, dtype="jax").full().flatten() # Initial state
# target_state = basis(3, 1, dtype="jax").full().flatten()  # Target for pi pulse

H0_mat = H_q_eff.full().astype(jnp.complex64)
drive_mat = b_q_eff.full().astype(jnp.complex64)
num_q_mat = num_q_q.full().astype(jnp.complex64)
num_q2_mat = num_q2_op.full().astype(jnp.complex64)

H_int = H0_mat

# Microwave pulse
pulse_duration = 70  # ns
n_segments = 10
segment_duration = pulse_duration / n_segments

H_drive = drive_mat

# Full time-dependent parametrized Hamiltonian
H = H_int + H_drive

# def evolve_states(y, H, params, t ):
#     amplitude, phase = params
#     def schrodinger_real_bin(y, t, H0_mat, drive_mat, amplitude, phase, omega_d_q):
#         psi_real = y[:3]
#         psi_imag = y[3:]
#         psi = psi_real + 1j * psi_imag
#         drive = amplitude * jnp.cos(omega_d_q * (t[1] + t[0]) + phase)  # Continue time
#         H = H0_mat + drive * drive_mat
#         dpsi_dt = -1j * jnp.dot(H, psi)
#         return jnp.concatenate([jnp.real(dpsi_dt), jnp.imag(dpsi_dt)])
    
#     t_bin = jnp.linspace(t[0], t[1], 10)  # Fine-grained for accuracy
#     return odeint(schrodinger_real_bin, y, t_bin, H0_mat, drive_mat, amplitude, omega_d_q)[-1]

from jax.experimental.ode import odeint
import jax
import jax.numpy as jnp

def evolve_states(y_batch, H, params_batch, t):
    # Unpack assuming H provides H0_mat and drive_mat (adjust if H is a dict or tuple)
    # H0_mat, drive_mat = H  # Or however you structure it; ensure they're jnp arrays
    
    def single_evolve(y, params, t):
        amplitude, phase = params
        
        def schrodinger_real_bin(y, t_val, H0_mat, drive_mat, amplitude, phase, omega_d_q):
            psi_real = y[:3]
            psi_imag = y[3:]
            psi = psi_real + 1j * psi_imag
            # Fixed: Use t_val (scalar) instead of t[1] + t[0]; assuming t is [start, end], but odeint passes scalars
            drive = amplitude * jnp.cos(omega_d_q * t_val + phase)
            H_t = H0_mat + drive * drive_mat
            dpsi_dt = -1j * jnp.dot(H_t, psi)
            return jnp.concatenate([jnp.real(dpsi_dt), jnp.imag(dpsi_dt)])
        
        t_bin = jnp.linspace(t[0], t[1], 10)  # Fine-grained for accuracy
        return odeint(schrodinger_real_bin, y, t_bin, H0_mat, drive_mat, amplitude, phase, omega_d_q)[-1]
    
    # Vectorize over batch (assumes params_batch shape matches y_batch's batch dim)
    vectorized_evolve = jax.vmap(single_evolve, in_axes=(0, 0, None))  # vmap over y and params, not t
    return vectorized_evolve(y_batch, params_batch, t)


state_size = 6

import jax.numpy as jnp

# jax.config.update("jax_enable_x64", True)  # Coment this line for a faster execution

values_phase = jnp.linspace(-jnp.pi, jnp.pi, 9)[1:]  # 8 phase values
values_ampl = jnp.linspace(0.0, 0.2, 11)  # 11 amplitude values
ctrl_values = jnp.stack(
    (jnp.repeat(values_ampl, len(values_phase)), jnp.tile(values_phase, len(values_ampl))), axis=1
)
n_actions = len(ctrl_values)  # 8x11 = 88 possible actions

from functools import partial

target = jnp.array([[0,1,0], [1, 0, 0], [0, 0, 1]])  # RX(pi/2) 


# @partial(jax.jit, static_argnames=["H", "config"])
@partial(jax.vmap, in_axes=(0, None, None, None, None))
def compute_rewards(pulse_params, H, target, config, subkey):
    """Compute the reward for the pulse program based on the average gate fidelity."""
    n_gate_reps = config.n_gate_reps
    # Sample the random initial states
    states = jnp.zeros((config.n_eval_states, n_gate_reps + 1, state_size), dtype=complex)
    states = states.at[:, 0, :].set(sample_random_states(subkey, config.n_eval_states, state_size))
    target_states = states.copy()

    # Repeatedly apply the gates and store the intermediate states
    print("pulse_params shape:", pulse_params.shape)
    print("states[:, 0] shape:", states[:, 0].shape)  # Should match batch_size
    
    time_window = (0, config.pulse_duration)
    for s in range(n_gate_reps):
        # states = states.at[:, s + 1].set(evolve_states(states[:, s], H, pulse_params, time_window)) 
        # target_states = target_states.at[:, s + 1].set(evolve_states(target_states[:, s],target, time_window))
        # Slice params for this segment to get (batch_size, n_params)
        params_for_segment = pulse_params[:, :, s]  # Adjust indices if dim order differs (e.g., [:, s, :])
        
        # Evolve main states
        evolved_states = evolve_states(states[:, s], H, params_for_segment, time_window)
        states = states.at[:, s + 1].set(evolved_states)
        
        # Evolve target states -- added missing pulse_params (assuming same as main; adjust if different)
        # If target is meant to be params, rename vars; assuming it's the target state tensor
        params_for_target = pulse_params[:, :, s]  # Or whatever params for target (e.g., ideal params)
        evolved_targets = evolve_states(target_states[:, s], H, params_for_target, time_window)
        target_states = target_states.at[:, s + 1].set(evolved_targets)
  


    # Compute all the state fidelities (excluding the initial states)
    overlaps = jnp.einsum("abc,abc->ab", target_states[:, 1:], jnp.conj(states[:, 1:]))
    fidelities = jnp.abs(overlaps) ** 2

    # Compute the weighted average gate fidelities
    weights = 2 * jnp.arange(n_gate_reps, 0, -1) / (n_gate_reps * (n_gate_reps + 1))
    rewards = jnp.einsum("ab,b->a", fidelities, weights)
    return rewards.mean()


@partial(jax.jit, static_argnames=["n_states", "dim"])
def sample_random_states(subkey, n_states, dim):
    """Sample random states from the Haar measure."""
    subkey0, subkey1 = jax.random.split(subkey, 2)

    s = jax.random.uniform(subkey0, (n_states, dim))
    s = -jnp.log(jnp.where(s == 0, 1.0, s))
    norm = jnp.sum(s, axis=-1, keepdims=True)
    phases = jax.random.uniform(subkey1, s.shape) * 2.0 * jnp.pi
    random_states = jnp.sqrt(s / norm) * jnp.exp(1j * phases)
    return random_states


# def get_pulse_matrix(H, params, time):
#     """Compute the unitary matrix associated to the time evolution of H."""
#     return qml.evolve(H)(params, time, atol=1e-5).matrix()


# @jax.jit
# def apply_gate(matrix, states):
#     """Apply the unitary matrix of the gate to a batch of states."""
#     return jnp.einsum("ab,cb->ca", matrix, states)

from flax import linen as nn


# Define the architecture
class MLP(nn.Module):
    """Multi layer perceptron (MLP) with a single hidden layer."""

    hidden_size: int
    out_size: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.tanh(x)
        x = nn.Dense(self.out_size)(x)
        return nn.softmax(jnp.sqrt((x * x.conj()).real))


policy_model = MLP(hidden_size=30, out_size=n_actions)

# Initialize the parameters passing a mock sample
key = jax.random.PRNGKey(3)
key, subkey = jax.random.split(key)

mock_state = jnp.empty((1, state_size))
policy_params = policy_model.init(subkey, mock_state)

# @partial(jax.jit, static_argnames=["H", "config"])
def play_episodes(policy_params, H, ctrl_values, target, config, key):
    """Play episodes in parallel."""
    n_episodes, n_segments = config.n_episodes, config.n_segments

    # Initialize the qubits on the |0> state
    states = jnp.zeros((n_episodes, n_segments + 1, state_size), dtype=complex)
    states = states.at[:, 0, 0].set(1.0)

    # Perform the PWC evolution of the pulse program
    pulse_params = jnp.zeros((n_episodes, 2, n_segments))
    actions = jnp.zeros((n_episodes, n_segments), dtype=int)
    score_functions = []
    for s in range(config.n_segments):
        # Observe the current state and select the parameters for the next pulse segment
        sf, (a, key) = act(states[:, s], policy_params, key)
        pulse_params = pulse_params.at[..., s].set(ctrl_values[a])
        print('pulse_params:', pulse_params)

        # Evolve the states with the next pulse segment
        time_window = (
            s * config.segment_duration,  # Start time
            (s + 1) * config.segment_duration,  # End time
        )
        states = states.at[:, s + 1].set(evolve_states(states[:, s], H, pulse_params, time_window))

        # Save the experience for posterior learning
        actions = actions.at[:, s].set(a)
        score_functions.append(sf)

    # Compute the final reward
    key, subkey = jax.random.split(key)
    rewards = compute_rewards(pulse_params, H, target, config, subkey)
    return states, actions, score_functions, rewards, key


@jax.jit
def act(states, params, key):
    """Act on states with the current policy params."""
    keys = jax.random.split(key, states.shape[0] + 1)
    score_funs, actions = score_function_and_action(params, states, keys[1:])
    return score_funs, (actions, keys[0])


@jax.jit
@partial(jax.vmap, in_axes=(None, 0, 0))
@partial(jax.grad, argnums=0, has_aux=True)
def score_function_and_action(params, state, subkey):
    """Sample an action and compute the associated score function."""
    probs = policy_model.apply(params, state)
    action = jax.random.choice(subkey, policy_model.out_size, p=probs)
    return jnp.log(probs[action]), action

@jax.jit
def sum_pytrees(pytrees):
    """Sum a list of pytrees."""
    return jax.tree_util.tree_map(lambda *x: sum(x), *pytrees)


@jax.jit
def adapt_shape(array, reference):
    """Adapts the shape of an array to match the reference (either a batched vector or matrix).
    Example:
    >>> a = jnp.ones(3)
    >>> b = jnp.ones((3, 2))
    >>> adapt_shape(a, b).shape
    (3, 1)
    >>> adapt_shape(a, b) + b
    Array([[2., 2.],
           [2., 2.],
           [2., 2.]], dtype=float32)
    """
    n_dims = len(reference.shape)
    if n_dims == 2:
        return array.reshape(-1, 1)
    return array.reshape(-1, 1, 1)


@jax.jit
def reinforce_gradient_with_baseline(episodes):
    """Estimates the parameter gradient from the episodes with a state-independent baseline."""
    _, _, score_functions, returns = episodes
    ret_episodes = returns.sum()  # Sum of episode returns to normalize the final value
    # b
    baseline = compute_baseline(episodes)
    # G - b
    ret_minus_baseline = jax.tree_util.tree_map(lambda b: adapt_shape(returns, b) - b, baseline)
    # sum((G - b) * sf)
    sf_sum = sum_pytrees(
        [jax.tree_util.tree_map(lambda r, s: r * s, ret_minus_baseline, sf) for sf in score_functions]
    )
    # E[sum((G - b) * sf)]
    return jax.tree_util.tree_map(lambda x: x.sum(0) / ret_episodes, sf_sum)


@jax.jit
def compute_baseline(episodes):
    """Computes the optimal state-independent baseline to minimize the gradient variance."""
    _, _, score_functions, returns = episodes
    n_episodes = returns.shape[0]
    n_segments = len(score_functions)
    total_actions = n_episodes * n_segments
    # Square of the score function: sf**2
    sq_sfs = jax.tree_util.tree_map(lambda sf: sf**2, score_functions)
    # Expected value: E[sf**2]
    exp_sq_sfs = jax.tree_util.tree_map(
        lambda sqsf: sqsf.sum(0, keepdims=True) / total_actions, sum_pytrees(sq_sfs)
    )
    # Return times score function squared: G*sf**2
    r_sq_sf = sum_pytrees(
        [jax.tree_util.tree_map(lambda sqsf: adapt_shape(returns, sqsf) * sqsf, sq_sf) for sq_sf in sq_sfs]
    )
    # Expected product: E[G_t*sf**2]
    exp_r_sq_sf = jax.tree_util.tree_map(lambda rsqsf: rsqsf.sum(0, keepdims=True) / total_actions, r_sq_sf)
    # Ratio of espectation values: E[G_t*sf**2] / E[sf**2]  (avoid dividing by zero)
    return jax.tree_util.tree_map(lambda ersq, esq: ersq / jnp.where(esq, esq, 1.0), exp_r_sq_sf, exp_sq_sfs)


import optax


def get_optimizer(params, learning_rate):
    """Create and initialize an Adam optimizer for the parameters."""
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)
    return optimizer, opt_state


def update_params(params, gradients, optimizer, opt_state):
    """Update model parameters with gradient ascent."""
    updates, opt_state = optimizer.update(gradients, opt_state, params)
    new_params = jax.tree_util.tree_map(lambda p, u: p - u, params, updates)  # Negative update
    return new_params, opt_state


from collections import namedtuple

hyperparams = [
    "pulse_duration",  # Total pulse duration
    "segment_duration",  # Duration of every pulse segment
    "n_segments",  # Number of pulse segments
    "n_episodes",  # Episodes to estimate the gradient
    "n_epochs",  # Training iterations
    "n_eval_states",  # Random states to evaluate the fidelity
    "n_gate_reps",  # Gate repetitions for the evaluation
    "learning_rate",  # Step size of the parameter update
]
Config = namedtuple("Config", hyperparams, defaults=[None] * len(hyperparams))

config = Config(
    pulse_duration=pulse_duration,
    segment_duration=segment_duration,
    n_segments=3,
    n_episodes=200,
    n_epochs=320,
    n_eval_states=10,
    n_gate_reps=1,
    learning_rate=5e-3,
)

optimizer, opt_state = get_optimizer(policy_params, config.learning_rate)

learning_rewards = []
for epoch in range(config.n_epochs):
    *episodes, key = play_episodes(policy_params, H, ctrl_values, target, config, key)
    grads = reinforce_gradient_with_baseline(episodes)
    policy_params, opt_state = update_params(policy_params, grads, optimizer, opt_state)

    learning_rewards.append(episodes[3].mean())
    if (epoch % 40 == 0) or (epoch == config.n_epochs - 1):
        print(f"Iteration {epoch}: reward {learning_rewards[-1]:.4f}")

import matplotlib.pyplot as plt

plt.plot(learning_rewards)
plt.xlabel("Training iteration")
plt.ylabel("Average reward")
plt.grid(alpha=0.3)