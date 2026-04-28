import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
from utils import CarDynamics, LPVController, plot_comparison


def mpc_loop(car: CarDynamics, ctrl: LPVController, 
             x0, u0, Ts, N_sim, reference_trajectory):
    """
    Main MPC loop.
    
    Args:
        car: CarDynamics instance
        ctrl: LPVController instance
        x0: initial state (6,)
        u0: initial input (2,)
        Ts: sampling time
        N_sim: number of simulation steps
        reference_trajectory: function(k) -> y_ref (4,) or array (N_sim, 4)
    
    Returns:
        X_history: (N_sim+1, 6) state trajectory
        U_history: (N_sim, 2) control inputs
        Y_history: (N_sim+1, 4) outputs
    """
    n_states = x0.shape[0]
    n_inputs = u0.shape[0]
    n_aug    = n_states + n_inputs
    
    X_history = np.zeros((N_sim + 1, n_states))
    U_history = np.zeros((N_sim, n_inputs))
    Y_history = np.zeros((N_sim + 1, 4))
    
    X_history[0] = x0
    u_prev = u0.copy()
    x_curr = x0.copy()
    
    # Get reference
    if callable(reference_trajectory):
        R_traj = np.array([reference_trajectory(k, Ts) for k in range(N_sim)])
    else:
        R_traj = reference_trajectory
    
    for k in range(N_sim):
        # 1. Build LTV trajectory from current state
        X_list, A_list, B_list, C_aug = ctrl.build_horizon_trajectory(
            x_curr, u_prev, Ts
        )
        
        # 2. Build time-varying matrices
        C_db, A_dh = ctrl.build_variable_matrices(A_list, B_list)
        
        # 3. Weight matrices (constant)
        n_out = C_aug.shape[0]
        Q = np.eye(n_out) * 2
        R = np.eye(n_inputs) * 0.1
        Q_db, T_db, R_db = ctrl.create_constant_db_matrices(Q, R, C_aug)
        
        # 4. H_db and F_db
        H_db = C_db.T @ Q_db @ C_db + R_db
        F_db = np.block([[A_dh.T @ Q_db @ C_db],
                         [-T_db @ C_db]]).T
        
        # 5. Augmented state
        x_aug_0 = np.zeros((n_aug, 1))
        x_aug_0[:n_states] = x_curr.reshape(-1, 1)
        x_aug_0[n_states:] = u_prev.reshape(-1, 1)
        
        # 6. Reference over horizon
        k_end = min(k + ctrl.horizon, N_sim)
        R_horizon = R_traj[k:k_end]
        if R_horizon.shape[0] < ctrl.horizon:
            # Pad with last reference if horizon exceeds simulation
            R_last = R_traj[-1]
            R_pad = np.tile(R_last, (ctrl.horizon - R_horizon.shape[0], 1))
            R_horizon = np.vstack([R_horizon, R_pad])
        R_ref = R_horizon.flatten().reshape(-1, 1)
        
        # 7. Solve MPC
        u_new, delta_U = ctrl.solve_mpc(H_db, F_db, x_aug_0, R_ref)
        u_new = u_new.flatten()
        
        # 8. Apply control and step system forward
        A, B, C, D = car.create_state_space_matrices(x_curr, u_new)
        Ad, Bd, _, _ = car.descrete_state_space(A, B, C, D, Ts)
        
        x_next = Ad @ x_curr + Bd @ u_new
        y_curr = C @ x_curr
        
        # 9. Store
        U_history[k] = u_new
        X_history[k + 1] = x_next
        Y_history[k] = y_curr
        
        # 10. Update for next iteration
        x_curr = x_next
        u_prev = u_new
        
        if k % 10 == 0:
            print(f"Step {k}/{N_sim}: x={np.round(x_curr[:3], 3)}, u={np.round(u_new, 3)}")
    
    # Final output
    Y_history[-1] = C @ X_history[-1]
    
    return X_history, U_history, Y_history


# ── Example: Straight line trajectory ────────────────────────────────────────
def straight_line_reference(k, Ts=0.1):
    """Reference: constant velocity, straight ahead"""
    vx_ref  = 10.0
    psi_ref = 0.0
    X_ref   = vx_ref * k * Ts
    Y_ref   = 0.0
    return np.array([vx_ref, psi_ref, X_ref, Y_ref])

def circle_reference(k, Ts=0.1, radius=20.0, speed=10.0):
    """Circular trajectory with constant speed"""
    t = k * Ts
    omega = speed / radius  # angular velocity
    
    vx_ref = speed
    psi_ref = omega * t
    X_ref = radius * np.sin(omega * t)
    Y_ref = radius * (1 - np.cos(omega * t))
    
    return np.array([vx_ref, psi_ref, X_ref, Y_ref])


def lane_change_reference(k, Ts=0.1, speed=15.0, lane_width=3.5, 
                          start_time=1.0, duration=3.0):
    """Double lane change maneuver"""
    t = k * Ts
    
    vx_ref = speed
    X_ref = speed * t
    
    # Smooth lane change using tanh
    if t < start_time:
        Y_ref = 0.0
        psi_ref = 0.0
    elif t < start_time + duration:
        progress = (t - start_time) / duration
        Y_ref = lane_width * np.tanh(4 * (progress - 0.5))
        # Approximate heading from Y derivative
        psi_ref = np.arctan2(lane_width * 4 / duration * (1 - np.tanh(4 * (progress - 0.5))**2), speed)
    else:
        Y_ref = lane_width
        psi_ref = 0.0
    
    return np.array([vx_ref, psi_ref, X_ref, Y_ref])


def figure_eight_reference(k, Ts=0.1, radius=15.0, speed=12.0):
    """Figure-8 trajectory"""
    t = k * Ts
    omega = speed / (2 * radius)
    
    vx_ref = speed
    X_ref = radius * np.sin(2 * omega * t)
    Y_ref = radius * np.sin(omega * t)
    
    # Compute heading from trajectory tangent
    dx = 2 * radius * omega * np.cos(2 * omega * t)
    dy = radius * omega * np.cos(omega * t)
    psi_ref = np.arctan2(dy, dx)
    
    return np.array([vx_ref, psi_ref, X_ref, Y_ref])


def slalom_reference(k, Ts=0.1, speed=10.0, amplitude=4.0, wavelength=30.0):
    """Slalom/sinusoidal trajectory"""
    t = k * Ts
    X_ref = speed * t
    
    freq = 2 * np.pi / wavelength
    Y_ref = amplitude * np.sin(freq * X_ref)
    
    # Heading from trajectory tangent
    dy_dx = amplitude * freq * np.cos(freq * X_ref)
    psi_ref = np.arctan(dy_dx)
    
    vx_ref = speed
    
    return np.array([vx_ref, psi_ref, X_ref, Y_ref])


def step_reference(k, Ts=0.1, speed=12.0, step_time=2.0, step_magnitude=5.0):
    """Step change in lateral position"""
    t = k * Ts
    
    vx_ref = speed
    X_ref = speed * t
    
    if t < step_time:
        Y_ref = 0.0
        psi_ref = 0.0
    else:
        Y_ref = step_magnitude
        psi_ref = 0.0
    
    return np.array([vx_ref, psi_ref, X_ref, Y_ref])


def acceleration_reference(k, Ts=0.1, v0=5.0, accel=2.0, max_speed=20.0):
    """Acceleration from low to high speed (straight line)"""
    t = k * Ts
    
    vx_ref = min(v0 + accel * t, max_speed)
    X_ref = v0 * t + 0.5 * accel * t**2 if vx_ref < max_speed else \
            v0 * t + 0.5 * accel * t**2  # Simplified
    Y_ref = 0.0
    psi_ref = 0.0
    
    return np.array([vx_ref, psi_ref, X_ref, Y_ref])


# ── Test suite runner ─────────────────────────────────────────────────────────
def run_trajectory_tests(car, horizon, Ts, test_configs):
    """
    Run MPC for multiple trajectory types and compare performance.
    
    Args:
        car: CarDynamics instance
        horizon: MPC horizon
        Ts: sampling time
        test_configs: list of dicts with keys:
            - 'name': trajectory name
            - 'ref_func': reference function
            - 'N_sim': simulation steps
            - 'x0': initial state
            - 'u0': initial input
    
    Returns:
        results: dict mapping trajectory name to (X_hist, U_hist, Y_hist, R_traj)
    """
    results = {}
    
    for config in test_configs:
        print(f"\n{'='*70}")
        print(f"Testing trajectory: {config['name']}")
        print(f"{'='*70}")
        
        ctrl = LPVController(horizon, car)
        
        X_hist, U_hist, Y_hist = mpc_loop(
            car, ctrl, 
            config['x0'], 
            config['u0'], 
            Ts, 
            config['N_sim'], 
            config['ref_func']
        )
        
        R_traj = np.array([config['ref_func'](k, Ts) for k in range(config['N_sim'])])
        
        results[config['name']] = {
            'X': X_hist,
            'U': U_hist,
            'Y': Y_hist,
            'R': R_traj
        }
    
    return results




def print_performance_table(results, Ts):
    """Print detailed performance metrics table"""
    print(f"\n{'='*90}")
    print(f"{'Trajectory':<20} {'Mean Err [m]':<15} {'Max Err [m]':<15} {'Mean |δ| [rad]':<20} {'Mean |Fx| [N]':<15}")
    print(f"{'='*90}")
    
    for name, data in results.items():
        Y_hist = data['Y']
        R_traj = data['R']
        U_hist = data['U']
        N = R_traj.shape[0]
        
        pos_error = np.sqrt((Y_hist[:N, 2] - R_traj[:, 2])**2 + 
                           (Y_hist[:N, 3] - R_traj[:, 3])**2)
        mean_err = np.mean(pos_error)
        max_err = np.max(pos_error)
        mean_delta = np.mean(np.abs(U_hist[:, 0]))
        mean_fx = np.mean(np.abs(U_hist[:, 1]))
        
        print(f"{name:<20} {mean_err:<15.4f} {max_err:<15.4f} {mean_delta:<20.4f} {mean_fx:<15.2f}")
    
    print(f"{'='*90}\n")


# ── Run all tests ─────────────────────────────────────────────────────────────
car = CarDynamics()
Ts = 0.1
horizon = 5

# states: [vx, vy, psi, pis_dot, X, Y]
test_configs = [
    {
        'name': 'Straight Line',
        'ref_func': straight_line_reference,
        'N_sim': 50,
        'x0': np.array([10.0, 0.1, 0.0, 0.0, 0.0, 0.0]),
        'u0': np.array([0.01, 1.0])
    },
    {
        'name': 'Circle',
        'ref_func': lambda k, Ts: circle_reference(k, Ts, radius=150.0, speed=5.0),
        'N_sim': 500,
        'x0': np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        'u0': np.array([0.1, 1.0])
    }
]

# Run tests
results = run_trajectory_tests(car, horizon, Ts, test_configs)

# Visualize
plot_comparison(results, Ts)
print_performance_table(results, Ts)





