import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import solve

class CarDynamics:
    def __init__(self):
        g=9.81; m=1500; Iz=3000
        Cf=38000; Cr=66000
        lf=2; lr=3; mu=0.02
        self.constants = {
            'g': g, 'm': m, 'Iz': Iz,
            'Cf': Cf, 'Cr': Cr,
            'lf': lf, 'lr': lr, 'mu': mu
        }

    def create_state_space_matrices(self, states, inputs):
        x_dot = states[0]
        y_dot = states[1]
        psi   = states[2]
        delta = inputs[0]

        g=self.constants['g'];   m=self.constants['m']
        Iz=self.constants['Iz']; Cf=self.constants['Cf']
        Cr=self.constants['Cr']; lf=self.constants['lf']
        lr=self.constants['lr']; mu=self.constants['mu']

        A11=-mu*g/x_dot
        A12= Cf*np.sin(delta)/(m*x_dot)
        A14= Cf*lf*np.sin(delta)/(m*x_dot) + y_dot
        A22=-(Cr + Cf*np.cos(delta))/(m*x_dot)
        A24=-(Cf*lf*np.cos(delta) - Cr*lr)/(m*x_dot) - x_dot
        A34= 1
        A42=-(Cf*lf*np.cos(delta) - lr*Cr)/(Iz*x_dot)
        A44=-(Cf*lf**2*np.cos(delta) + lr**2*Cr)/(Iz*x_dot)
        A51= np.cos(psi);  A52=-np.sin(psi)
        A61= np.sin(psi);  A62= np.cos(psi)

        B11=-Cf*np.sin(delta)/m
        B12= 1.0
        B21= Cf*np.cos(delta)/m
        B41= Cf*lf*np.cos(delta)/Iz

        A = np.array([[A11, A12, 0,   A14, 0, 0],
                      [0,   A22, 0,   A24, 0, 0],
                      [0,   0,   0,   A34, 0, 0],
                      [0,   A42, 0,   A44, 0, 0],
                      [A51, A52, 0,   0,   0, 0],
                      [A61, A62, 0,   0,   0, 0]])

        B = np.array([[B11, B12],
                      [B21, 0  ],
                      [0,   0  ],
                      [B41, 0  ],
                      [0,   0  ],
                      [0,   0  ]])

        C = np.array([[1,0,0,0,0,0],
                      [0,0,1,0,0,0],
                      [0,0,0,0,1,0],
                      [0,0,0,0,0,1]])

        D = np.zeros((4, 2))
        return A, B, C, D

    def descrete_state_space(self, A, B, C, D, Ts: float):
        I     = np.eye(A.shape[0])
        A_des = I + A * Ts
        B_des = B * Ts
        return A_des, B_des, C, D

    def augment_states(self, A_des, B_des, C_des):
        n  = A_des.shape[0]
        nu = B_des.shape[1]

        A_aug = np.block([[A_des,              B_des      ],
                          [np.zeros((nu, n)),  np.eye(nu) ]])

        B_aug = np.block([[B_des     ],
                          [np.eye(nu)]])

        C_aug = np.block([C_des, np.zeros((C_des.shape[0], nu))])

        return A_aug, B_aug, C_aug

class LPVController:
    def __init__(self, horizon, car: CarDynamics):
        self.horizon = horizon
        self.car     = car

    def build_horizon_trajectory(self, x0, u0, Ts):
        X_list    = [x0.copy()]
        A_list    = []
        B_list    = []
        C_aug_out = None

        x = x0.copy()
        u = u0.copy()

        for k in range(self.horizon):
            A, B, C, D          = self.car.create_state_space_matrices(x, u)
            Ad, Bd, Cd, Dd      = self.car.descrete_state_space(A, B, C, D, Ts)
            A_aug, B_aug, C_aug = self.car.augment_states(Ad, Bd, Cd)

            A_list.append(A_aug)
            B_list.append(B_aug)

            if k == 0:
                C_aug_out = C_aug

            x = Ad @ x + Bd @ u
            X_list.append(x)

        return X_list, A_list, B_list, C_aug_out

    def build_variable_matrices(self, A_tilde_list, B_tilde_list):
        n = B_tilde_list[0].shape[0]
        m = B_tilde_list[0].shape[1]
        h = self.horizon

        C_db = np.zeros((h * n, h * m))
        A_dh = np.zeros((h * n, n))

        for h_col in range(h):
            acum = B_tilde_list[h_col]
            for h_row in range(h_col, h):
                rs = h_row * n;  cs = h_col * m
                C_db[rs:rs+n, cs:cs+m] = acum
                if h_row + 1 < h:
                    acum = A_tilde_list[h_row + 1] @ acum

        for h_row in range(h):
            rs = h_row * n
            if h_row == 0:
                acum = A_tilde_list[0]
            else:
                acum = A_tilde_list[h_row] @ acum
            A_dh[rs:rs+n, :] = acum

        return C_db, A_dh

    def create_constant_db_matrices(self, Q, R, C_tilde):
        diag_Q = C_tilde.T @ Q @ C_tilde
        Q_db   = np.kron(np.eye(self.horizon), diag_Q)
        T_db   = np.kron(np.eye(self.horizon), Q @ C_tilde)
        R_db   = np.kron(np.eye(self.horizon), R)
        return Q_db, T_db, R_db

    def solve_mpc(self, H_db, F_db, x_aug_0, R_ref):
        """
        delta_U* = -inv(H_db) @ F_db @ [x_aug_0; R_ref]
        u_new    = u_prev + delta_U*[0:n_in]

        x_aug_0 : (n_aug, 1)
        R_ref   : (h * n_out, 1)  — stacked reference over horizon
        """
        n_in  = H_db.shape[0] // self.horizon

        z       = np.vstack([x_aug_0, R_ref])           # (n_aug + h*n_out, 1)
        delta_U = -solve(H_db, F_db @ z)              # (h*n_in, 1)
        delta_u0 = delta_U[:n_in]                        # first increment only
        n_aug    = x_aug_0.shape[0]
        u_prev   = x_aug_0[n_aug - n_in:]               # u lives at end of aug state
        u_new    = u_prev + delta_u0

        return u_new, delta_U

def plot_mpc_results(X_hist, U_hist, Y_hist, R_traj, Ts):
    """
    Plot detailed MPC results with reference vs actual for each signal.
    
    Args:
        X_hist: (N_sim+1, n_states) state history
        U_hist: (N_sim, n_inputs) control history
        Y_hist: (N_sim, n_outputs) output history
        R_traj: (N_sim, n_outputs) reference trajectory
        Ts: sampling time
    """
    N_sim = R_traj.shape[0]
    time_output = np.arange(N_sim) * Ts
    time_state = np.arange(N_sim + 1) * Ts
    time_control = np.arange(N_sim) * Ts
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # ── X-Y Trajectory ────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(R_traj[:, 2], R_traj[:, 3], '--', color='red', linewidth=2, 
             label='Reference', alpha=0.7)
    ax1.plot(Y_hist[:, 2], Y_hist[:, 3], '-', color='blue', linewidth=2, 
             label='Actual')
    ax1.scatter(Y_hist[0, 2], Y_hist[0, 3], color='green', s=100, 
                marker='o', label='Start', zorder=5)
    ax1.scatter(Y_hist[-1, 2], Y_hist[-1, 3], color='red', s=100, 
                marker='s', label='End', zorder=5)
    ax1.set_xlabel('$X$ [m]', fontsize=12)
    ax1.set_ylabel('$Y$ [m]', fontsize=12)
    ax1.set_title('X-Y Trajectory', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # ── X Position vs Time ────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(time_output, R_traj[:, 2], '--', color='red', linewidth=2, 
             label='$X_{ref}$', alpha=0.7)
    ax2.plot(time_output, Y_hist[:, 2], '-', color='blue', linewidth=2, 
             label='$X$')
    ax2.set_xlabel('Time [s]', fontsize=11)
    ax2.set_ylabel('$X$ [m]', fontsize=11)
    ax2.set_title('Longitudinal Position', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # ── Y Position vs Time ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(time_output, R_traj[:, 3], '--', color='red', linewidth=2, 
             label='$Y_{ref}$', alpha=0.7)
    ax3.plot(time_output, Y_hist[:, 3], '-', color='blue', linewidth=2, 
             label='$Y$')
    ax3.set_xlabel('Time [s]', fontsize=11)
    ax3.set_ylabel('$Y$ [m]', fontsize=11)
    ax3.set_title('Lateral Position', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # ── Velocity vs Time ──────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(time_output, R_traj[:, 0], '--', color='red', linewidth=2, 
             label='$v_{x,ref}$', alpha=0.7)
    ax4.plot(time_state, X_hist[:, 0], '-', color='blue', linewidth=2, 
             label='$v_x$')
    ax4.set_xlabel('Time [s]', fontsize=11)
    ax4.set_ylabel('$v_x$ [m/s]', fontsize=11)
    ax4.set_title('Longitudinal Velocity', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # ── Heading Angle vs Time ─────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(time_output, R_traj[:, 1], '--', color='red', linewidth=2, 
             label='$\psi_{ref}$', alpha=0.7)
    ax5.plot(time_output, Y_hist[:, 1], '-', color='blue', linewidth=2, 
             label='$\psi$')
    ax5.set_xlabel('Time [s]', fontsize=11)
    ax5.set_ylabel('$\psi$ [rad]', fontsize=11)
    ax5.set_title('Heading Angle', fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # ── Steering Angle ────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.step(time_control, U_hist[:, 0], '-', color='purple', linewidth=2, 
             where='post', label='$\delta$')
    ax6.set_xlabel('Time [s]', fontsize=11)
    ax6.set_ylabel('$\delta$ [rad]', fontsize=11)
    ax6.set_title('Steering Angle', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # ── Longitudinal Force ────────────────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.step(time_control, U_hist[:, 1], '-', color='orange', linewidth=2, 
             where='post', label='$F_x$')
    ax7.set_xlabel('Time [s]', fontsize=11)
    ax7.set_ylabel('$F_x$ [N]', fontsize=11)
    ax7.set_title('Longitudinal Force', fontsize=12, fontweight='bold')
    ax7.legend(fontsize=9)
    ax7.grid(True, alpha=0.3)
    
    plt.savefig('mpc_detailed_results.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ── Performance Metrics ───────────────────────────────────────────────────
    pos_error = np.sqrt((Y_hist[:, 2] - R_traj[:, 2])**2 + 
                       (Y_hist[:, 3] - R_traj[:, 3])**2)
    vx_error = np.abs(X_hist[:N_sim, 0] - R_traj[:, 0])
    psi_error = np.abs(Y_hist[:, 1] - R_traj[:, 1])
    
    print(f"\n{'='*60}")
    print(f"Performance Metrics:")
    print(f"{'='*60}")
    print(f"Position Error:  Mean = {np.mean(pos_error):.4f} m,  Max = {np.max(pos_error):.4f} m")
    print(f"Velocity Error:  Mean = {np.mean(vx_error):.4f} m/s, Max = {np.max(vx_error):.4f} m/s")
    print(f"Heading Error:   Mean = {np.mean(psi_error):.4f} rad, Max = {np.max(psi_error):.4f} rad")
    print(f"Control Effort:  Mean |δ| = {np.mean(np.abs(U_hist[:, 0])):.4f} rad")
    print(f"                 Mean |Fx| = {np.mean(np.abs(U_hist[:, 1])):.2f} N")
    print(f"{'='*60}\n")

def plot_comparison(results, Ts: float):
    """
    Create separate detailed plots for each trajectory.
    
    Parameters
    ----------
    results : dict
        Dictionary mapping trajectory names to result dictionaries from mpc_loop.
        Each result dict contains 'X', 'Y', 'U', 'R'.
    Ts : float
        Sampling time [s].
    """
    for name, data in results.items():
        Y_hist = data['Y']   # (N+1, n_out)
        R_traj = data['R']   # (N, n_out)
        U_hist = data['U']   # (N, n_in)
        N = R_traj.shape[0]
        time = np.arange(N) * Ts
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 12))
        fig.suptitle(f'MPC Results: {name}', fontsize=16, fontweight='bold')
        
        # ── X-Y Trajectory ────────────────────────────────────────────────
        ax = axes[0, 0]
        ax.plot(R_traj[:, 2], R_traj[:, 3], 'b-', linewidth=2, label='Reference')
        ax.plot(Y_hist[:N, 2], Y_hist[:N, 3], 'r--', linewidth=2, label='Actual')
        ax.set_xlabel('X Position [m]')
        ax.set_ylabel('Y Position [m]')
        ax.set_title('X-Y Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # ── X Position vs Time ────────────────────────────────────────────
        ax = axes[0, 1]
        ax.plot(time, R_traj[:, 2], 'b-', linewidth=2, label='Reference')
        ax.plot(time, Y_hist[:N, 2], 'r--', linewidth=2, label='Actual')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('X Position [m]')
        ax.set_title('X Position')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ── Y Position vs Time ────────────────────────────────────────────
        ax = axes[0, 2]
        ax.plot(time, R_traj[:, 3], 'b-', linewidth=2, label='Reference')
        ax.plot(time, Y_hist[:N, 3], 'r--', linewidth=2, label='Actual')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Y Position [m]')
        ax.set_title('Y Position')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ── Longitudinal Velocity vs Time ─────────────────────────────────
        ax = axes[1, 0]
        ax.plot(time, R_traj[:, 0], 'b-', linewidth=2, label='Reference')
        ax.plot(time, Y_hist[:N, 0], 'r--', linewidth=2, label='Actual')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('$v_x$ [m/s]')
        ax.set_title('Longitudinal Velocity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ── Heading Angle vs Time ─────────────────────────────────────────
        ax = axes[1, 1]
        ax.plot(time, R_traj[:, 1], 'b-', linewidth=2, label='Reference')
        ax.plot(time, Y_hist[:N, 1], 'r--', linewidth=2, label='Actual')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('$\\psi$ [rad]')
        ax.set_title('Heading Angle')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # ── Position Error vs Time ────────────────────────────────────────
        ax = axes[1, 2]
        pos_error = np.sqrt((Y_hist[:N, 2] - R_traj[:, 2])**2 + 
                           (Y_hist[:N, 3] - R_traj[:, 3])**2)
        ax.plot(time, pos_error, 'k-', linewidth=2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Position Error [m]')
        ax.set_title('Position Error')
        ax.grid(True, alpha=0.3)
        
        # ── Steering Angle vs Time ────────────────────────────────────────
        ax = axes[2, 0]
        ax.plot(time, U_hist[:, 0], 'g-', linewidth=2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('$\\delta$ [rad]')
        ax.set_title('Steering Angle')
        ax.grid(True, alpha=0.3)
        
        # ── Longitudinal Force vs Time ────────────────────────────────────
        ax = axes[2, 1]
        ax.plot(time, U_hist[:, 1], 'm-', linewidth=2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('$F_x$ [N]')
        ax.set_title('Longitudinal Force')
        ax.grid(True, alpha=0.3)
        
        # ── Performance Metrics ───────────────────────────────────────────
        ax = axes[2, 2]
        ax.axis('off')
        
        mean_pos_error = np.mean(pos_error)
        max_pos_error = np.max(pos_error)
        mean_steering = np.mean(np.abs(U_hist[:, 0]))
        mean_force = np.mean(np.abs(U_hist[:, 1]))
        
        metrics_text = f"""
        Performance Metrics:
        
        Mean Position Error: {mean_pos_error:.4f} m
        Max Position Error:  {max_pos_error:.4f} m
        
        Mean |δ|:  {mean_steering:.4f} rad
        Mean |Fx|: {mean_force:.2f} N
        """
        
        ax.text(0.1, 0.5, metrics_text, fontsize=12, 
                verticalalignment='center', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        
        plt.tight_layout()
        plt.show()
