import numpy as np
import torch
import torch.nn as nn
import casadi as ca
import matplotlib.pyplot as plt
import pickle
import os
try:
    from train import MLP
except ImportError:
    from src.train import MLP

class MPCConfig:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    EXP_FOLDER = "joined"
    MODEL_PATH = os.path.join(BASE_DIR, "..", "data", EXP_FOLDER, "best_model.pth")
    SCALERS_PATH = os.path.join(BASE_DIR, "..", "data", EXP_FOLDER, "scalers.pkl")
    
    N = 3  # MPC horizon
    DT = 0.001  # time step
    SIM_TIME = 10.0
    
    # Cost weights
    Q_pos = 100000000.0  # position tracking weight
    R_control = 0.0  # control effort weight
    R_rate = 10.0  # control rate weight
    LAMBDA = 200.0  # terminal cost weight
    DZ_COST = 2000.0  # dead zone penalty weight
    SIGMA = 60.0  # dead zone penalty sharpness
    
    # Control bounds (PWM values)
    U_MIN = np.array([-255, -255, -255, -255])
    U_MAX = np.array([255, 255, 255, 255])
    U_DEADZONE_MIN = np.array([-150, -150, -150, -150])
    U_DEADZONE_MAX = np.array([150, 150, 150, 150])
    
    # Model configuration
    USE_NONLINEAR_MODEL = False

class MPCController:
    def __init__(self):
        self.n_outputs = 2  # px, py
        self.n_controls = 4  # pm1, pm2, pm3, pm4
        
        # Load model and scalers
        self._load_assets()
        self._setup_optimization_problem()
        
        # Initialize history
        self.history_y = []
        self.history_u = []
        # self.u_previous = np.zeros(self.n_controls)  # Store previous control for rate penalty
        self.u_previous = np.array([150, 150, -150, -150])  # Start from a nominal point away from deadzone
        
        print(f"Initialized MPCController")
        print(f"  Outputs: {self.n_outputs}, Controls: {self.n_controls}")
        print(f"  Horizon N: {MPCConfig.N}, Time step: {MPCConfig.DT}")
        print(f"  Nonlinear Model: {MPCConfig.USE_NONLINEAR_MODEL}")
        
    def _load_assets(self):
        """Load model and scalers"""
        print("Loading model and scalers...")
        
        # Load scalers
        with open(MPCConfig.SCALERS_PATH, 'rb') as f:
            scalers_data = pickle.load(f)
        
        self.x_scaler = scalers_data['x_scaler']
        self.y_scaler = scalers_data['y_scaler']
        self.input_cols = scalers_data['input_cols']
        self.output_cols = scalers_data['output_cols']
        
        # Load model
        checkpoint = torch.load(MPCConfig.MODEL_PATH, map_location='cpu', weights_only=True)
        if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        else:
            state_dict = checkpoint
            
        # Create model with correct dimensions
        in_dim = len(self.input_cols)
        out_dim = len(self.output_cols)
        self.model = MLP(in_dim=in_dim, out_dim=out_dim)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        print(f"Loaded model: {in_dim} inputs -> {out_dim} outputs")
        print(f"Input columns: {self.input_cols}")
        print(f"Output columns: {self.output_cols}")
    
    def _model_prediction_casadi(self, u_in):
        """
        Symbolic translation of the PyTorch MLP to CasADi.
        u_in: CasADi expression (n_controls, 1)
        Returns: CasADi expression (n_outputs, 1)
        """
        # 1. Scale input
        # x_scaled = (x - mean) / scale
        x_mean = self.x_scaler.mean_
        x_scale = self.x_scaler.scale_
        
        x = (u_in - x_mean.reshape(-1,1)) / x_scale.reshape(-1,1)
        
        # 2. Forward pass through network layers
        # Access weights from self.model.network. The structure is:
        # 0: Linear(in, 128)
        # 1: ReLU
        # 2: Linear(128, 256)
        # 3: ReLU
        # 4: Linear(256, 128)
        # 5: ReLU
        # 6: Linear(128, out)
        
        layers = [
            (self.model.network[0], True),   # True for ReLU activation
            (self.model.network[2], True),
            (self.model.network[4], True),
            (self.model.network[6], False)  # No activation on output
        ]
        
        for layer, use_relu in layers:
            W = layer.weight.detach().numpy()
            b = layer.bias.detach().numpy()
            
            # Linear transform: Wx + b
            x = ca.mtimes(W, x) + b.reshape(-1, 1)
            
            # Activation
            if use_relu:
                # Smooth ReLU (Softplus) approximation for better IPOPT convergence
                # f(x) = log(1 + exp(x))
                # To avoid overflow, use equivalent form for large x
                # For basic stability: ca.log(1 + ca.exp(x))
                x = ca.log(1 + ca.exp(x))
                
        # 3. Inverse scale output
        # y = y_scaled * scale + mean
        y_scaled = x
        y_mean = self.y_scaler.mean_
        y_scale = self.y_scaler.scale_
        
        y = y_scaled * y_scale.reshape(-1, 1) + y_mean.reshape(-1, 1)
        
        return y

    def _setup_optimization_problem(self):
        """Setup CasADi optimization problem"""
        self.opti = ca.Opti()
        
        # Decision variables
        self.U = self.opti.variable(self.n_controls, MPCConfig.N)  # Control sequence
        
        # Parameters
        self.y_ref = self.opti.parameter(self.n_outputs, 1)  # Reference output
        self.lambda_param = self.opti.parameter(1, 1)  # Terminal cost weight
        self.P_terminal = self.opti.parameter(self.n_outputs, self.n_outputs)  # Terminal cost matrix
        self.u_prev = self.opti.parameter(self.n_controls, 1)  # Previous control input
        
        # Linear approximation parameters (still used for terminal cost and possibly tracking if flag is False)
        self.A_lin = self.opti.parameter(self.n_outputs, self.n_controls)
        self.b_lin = self.opti.parameter(self.n_outputs, 1)
        
        # Cost function
        cost = 0
        for k in range(MPCConfig.N):
            u_k = self.U[:, k]
            
            # Predict output
            if MPCConfig.USE_NONLINEAR_MODEL:
                # Use the full neural network symbolic graph
                y_k = self._model_prediction_casadi(u_k)
            else:
                # Use linear approximation
                y_k = self.A_lin @ u_k + self.b_lin
            
            # Tracking cost
            cost += MPCConfig.Q_pos * ca.sumsqr(y_k - self.y_ref)
            
            # Control effort cost
            cost += MPCConfig.R_control * ca.sumsqr(u_k)
            
            # Control rate cost
            if k == 0:
                cost += MPCConfig.R_rate * ca.sumsqr(u_k - self.u_prev)
            else:
                cost += MPCConfig.R_rate * ca.sumsqr(u_k - self.U[:, k-1])

            # Dead zone cost
            cost += MPCConfig.DZ_COST * ca.exp(-(u_k[0]**2)/MPCConfig.SIGMA)
            cost += MPCConfig.DZ_COST * ca.exp(-(u_k[1]**2)/MPCConfig.SIGMA)
            cost += MPCConfig.DZ_COST * ca.exp(-(u_k[2]**2)/MPCConfig.SIGMA)
            cost += MPCConfig.DZ_COST * ca.exp(-(u_k[3]**2)/MPCConfig.SIGMA)

        
        # Terminal cost
        if MPCConfig.USE_NONLINEAR_MODEL:
             y_terminal = self._model_prediction_casadi(self.U[:, -1])
        else:
             y_terminal = self.A_lin @ self.U[:, -1] + self.b_lin
             
        y_terminal_error = y_terminal - self.y_ref
        cost += self.lambda_param * y_terminal_error.T @ self.P_terminal @ y_terminal_error
        
        self.opti.minimize(cost)
        
        # Control constraints
        for k in range(MPCConfig.N):
            self.opti.subject_to(self.opti.bounded(MPCConfig.U_MIN, self.U[:, k], MPCConfig.U_MAX))

        # Solver settings
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0, 
            'ipopt.sb': 'yes',
            # 'ipopt.max_iter': 1000,        
            # 'ipopt.tol': 1e-4,           
            # 'ipopt.accept_after_max_steps': 10000,
            # 'ipopt.hessian_approximation': 'limited-memory'
        }
        self.opti.solver('ipopt', opts)
    
    def predict_output(self, u_input):
        """Predict output using neural network"""
        u_scaled = self.x_scaler.transform(u_input.reshape(1, -1))
        u_tensor = torch.from_numpy(u_scaled).float()
        
        with torch.no_grad():
            y_scaled = self.model(u_tensor)
            y_pred = self.y_scaler.inverse_transform(y_scaled.numpy())
        
        return y_pred.flatten()
    
    def linearize_model(self, u_nom):
        """Compute linear approximation of model around nominal point"""
        u_tensor = torch.from_numpy(self.x_scaler.transform(u_nom.reshape(1, -1))).float()
        u_tensor.requires_grad_(True)
        
        # Forward pass
        y_scaled = self.model(u_tensor)
        
        # Compute full Jacobian using torch.autograd.grad
        jacobian = torch.zeros(self.n_outputs, self.n_controls)
        
        for i in range(self.n_outputs):
            # Create fresh computation graph for each output
            u_fresh = torch.from_numpy(self.x_scaler.transform(u_nom.reshape(1, -1))).float()
            u_fresh.requires_grad_(True)
            y_fresh = self.model(u_fresh)
            
            # Compute gradient for this output
            grad_inputs = torch.autograd.grad(outputs=y_fresh[0, i], 
                                            inputs=u_fresh, 
                                            retain_graph=False, 
                                            create_graph=False)[0]
            jacobian[i, :] = grad_inputs[0, :]
        
        # Scale jacobian back to original units
        A_scaled = jacobian.detach().numpy()
        A_original = A_scaled * (self.y_scaler.scale_.reshape(-1, 1) / self.x_scaler.scale_.reshape(1, -1))
        
        # Compute bias term
        y_pred = self.y_scaler.inverse_transform(y_scaled.detach().numpy()).flatten()
        b_original = y_pred - A_original @ u_nom
        
        return A_original, b_original
    
    def compute_terminal_cost_matrix(self, A_lin):
        """Compute terminal cost matrix for the linearized system"""
        try:
            # For the simple case y = A*u + b, we use a simple identity-based terminal cost
            # In practice, you might want to solve a discrete algebraic Riccati equation
            # For now, use identity matrix scaled appropriately
            P_terminal = MPCConfig.Q_pos * np.eye(self.n_outputs)
            return P_terminal
        except Exception as e:
            print(f"Warning: Failed to compute terminal cost matrix: {e}")
            return MPCConfig.Q_pos * np.eye(self.n_outputs)
    
    def step(self, y_ref, y_current=None):
        """Solve MPC and return optimal control input"""
        # Use previous control as nominal point for linearization
        if hasattr(self, 'u_previous'):
            u_nom = self.u_previous.copy()
        else:
            u_nom = np.zeros(self.n_controls)
        
        # Linearize model around nominal point
        A_lin, b_lin = self.linearize_model(u_nom)
        
        # If current position is provided, adjust the bias term to match reality
        if y_current is not None:
            # Adjust bias to make the linear model match the current measurement
            y_pred_nom = A_lin @ u_nom + b_lin
            error = y_current - y_pred_nom
            b_lin = b_lin + error  # Correct the bias term
        
        # Compute terminal cost matrix
        P_terminal = self.compute_terminal_cost_matrix(A_lin)
        lambda_weight = MPCConfig.LAMBDA
        
        # Set parameters
        self.opti.set_value(self.y_ref, y_ref.reshape(-1, 1))
        self.opti.set_value(self.A_lin, A_lin)
        self.opti.set_value(self.b_lin, b_lin.reshape(-1, 1))
        self.opti.set_value(self.P_terminal, P_terminal)
        self.opti.set_value(self.lambda_param, np.array([[lambda_weight]]))
        self.opti.set_value(self.u_prev, self.u_previous.reshape(-1, 1))
        
        try:
            sol = self.opti.solve()
            u_optimal = sol.value(self.U[:, 0])  # Take first control action
            self.u_previous = u_optimal.copy()  # Store for next iteration rate penalty
            return u_optimal
        except Exception as e:
            print(f"MPC solver failed: {e}")
            # Try to get debug value if available
            try:
                debug_u = self.opti.debug.value(self.U[:, 0])
                print("Returning debug value.")
                self.u_previous = debug_u.copy()
                return debug_u
            except:
                 pass
            
            return u_nom  # Return nominal control instead of zeros
    
    def plot_results(self, smooth=True):
        """Plot MPC results"""
        if len(self.history_y) == 0:
            print("No data to plot")
            return
            
        history_y = np.array(self.history_y)
        history_u = np.array(self.history_u)

        if smooth:
            from scipy.ndimage import gaussian_filter1d
            history_y = gaussian_filter1d(history_y, sigma=2, axis=0)
            history_u = gaussian_filter1d(history_u, sigma=2, axis=0)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        time_axis = np.arange(len(history_y)) * MPCConfig.DT
        
        plt.rcParams.update({'font.size': 14})
        
        # Output plot
        ax1.plot(time_axis, history_y[:, 0], label='px')
        ax1.plot(time_axis, history_y[:, 1], label='py')
        ax1.set_ylabel('Output [mm]')
        ax1.legend()
        ax1.grid(True)
        ax1.set_title('MPC Tracking Performance')
        
        # Control plot
        time_axis_u = np.arange(len(history_u)) * MPCConfig.DT
        ax2.plot(time_axis_u, history_u[:, 0], label='pm1')
        ax2.plot(time_axis_u, history_u[:, 1], label='pm2')
        ax2.plot(time_axis_u, history_u[:, 2], label='pm3')
        ax2.plot(time_axis_u, history_u[:, 3], label='pm4')
        ax2.set_ylabel('Control [PWM]')
        ax2.set_xlabel('Time [s]')
        ax2.legend()
        ax2.grid(True)
        ax2.set_title('Control Inputs')
        
        plt.tight_layout()
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/mpc_results.png')
        print("Results saved to results/mpc_results.png")
        plt.show()

def run_mpc_simulation():
    """Run MPC simulation"""
    mpc = MPCController()
    
    print("Starting MPC simulation...")
    
    # Set reference trajectory (step input)
    target = np.array([1.0, 0.5])  # Target px, py in mm
    print(f"Target position: {target}")
    
    n_steps = int(MPCConfig.SIM_TIME / MPCConfig.DT)
    
    for i in range(n_steps):
        # Get MPC control
        u_optimal = mpc.step(target)
        
        # Simulate system response
        y_current = mpc.predict_output(u_optimal)
        
        # Store data
        mpc.history_y.append(y_current)
        mpc.history_u.append(u_optimal)
        
        if i % 10 == 0:
            error = np.linalg.norm(y_current - target)
            print(f"Step {i+1}/{n_steps}: Control = {u_optimal}, Output = {y_current}, Error = {error:.4f}")
    
    # Plot results
    mpc.plot_results()

if __name__ == "__main__":
    run_mpc_simulation()