"""
Nonlinear state space model algorithms
"""


#######################################################
# Modules:
from numpy import matmul, add, subtract, transpose, zeros
from scipy.linalg import inv
from tqdm import tqdm

# Local modules:
from basic_tools import get_kwarg_value


#######################################################
# Extended Kalman filter algorithm class (with options to include BAE):
# --------------- Assumptions ---------------
# 1. w_k ~ N(0, Gamma_w) is a stationary noise process.
# 2. v_k ~ N(0, Gamma_v_k) is a time-varying noise process.
# 3. Cross covariances involving BAE are zero.
# --------------- Output ---------------
# Predition step: x_{k|k-1}, Gamma_{k|k-1| = predict(x_{k-1|k-1}, Gamma_{k-1|k-1})
# Update step: x_{k|k}, Gamma_{k|k} = update(x_{k|k-1}, Gamma_{k|k-1}, y_k)
# --------------- Class ---------------
class Extended_Kalman_filter:

    def __init__(self, F, J_F, H, J_H, Gamma_w, Gamma_v, NT, **kwargs):
        N = len(Gamma_w[0])  # Dimensions of state vector
        M = len(Gamma_v[0])  # Dimensions of observation vector
        # Bayesian Approximation Error parameters:
        NT_obs = get_kwarg_value(kwargs, 'NT_obs', NT)  # Total number of time steps for observations
        self.mean_epsilon = get_kwarg_value(kwargs, 'mean_epsilon', zeros([N, NT]))  # Option to include BAE for evolution model
        self.Gamma_epsilon = get_kwarg_value(kwargs, 'Gamma_epsilon', zeros([NT, N, N]))  # Option to include BAE for evolution model
        self.mean_nu = get_kwarg_value(kwargs, 'mean_nu', zeros([M, NT_obs]))  # Option to include BAE for observation model
        self.Gamma_nu = get_kwarg_value(kwargs, 'Gamma_nu', zeros([NT_obs, M, M]))  # Option to include BAE for observation model
        # Additive vectors to model:
        self.additive_evolution_vector = get_kwarg_value(kwargs, 'additive_evolution_vector', zeros(N))  # Additive vector to evolution model
        # Operators:
        self.F = F  # Evolution operator F = F(x, t)
        self.J_F = J_F  # Evolution operator Jacobian J_F = J_F(x, t)
        self.H = H  # Observation operator H = H(x, t)
        self.J_H = J_H  # Observation operator Jacobian J_H = J_H(x, t)
        # Covariances:
        self.Gamma_w = Gamma_w  # Evolution noise covariance matrix; w_k ~ N(0, Gamma_w)
        self.Gamma_v = Gamma_v  # Observation noise covariance matrix; v_k ~ N(0, Gamma_v_k)
        # Class properties for computation purposes:
        self.J_Hx = 0  # Initialising evaluated observation operator Jacobian (for computation purposes)
        self.J_HxT = 0  # Initialising evaluated observation operator Jacobian (transposed) (for computation purposes)

    def compute_Gamma_predict(self, x, t, Gamma):
        J_Fx = self.J_F(x, t)  # Evolution Jacobian evaluation at x_{t-1,t-1}
        J_FxT = transpose(J_Fx)  # Transpose of Jacobian
        M1 = matmul(Gamma, J_FxT)  # Matrix operation
        M2 = matmul(J_Fx, M1)  # Matrix operation
        output = add(self.Gamma_w, M2)  # Matrix operation
        return output

    def compute_Gamma_innovation(self, x_predict, t, k, Gamma_predict):
        self.J_Hx = self.J_H(x_predict, t)  # Observation Jacobian evaluation at x_{t,t-1}
        self.J_HxT = transpose(self.J_Hx)  # Transpose of Jacobian
        M1 = matmul(Gamma_predict, self.J_HxT)  # Matrix operation
        M2 = matmul(self.J_Hx, M1)  # Matrix operation
        output = add(self.Gamma_v[k], M2)  # Matrix operation
        return output

    def compute_Kalman_gain(self, Gamma_predict, Gamma_innovation):
        M1 = inv(Gamma_innovation)  # Matrix operation
        M2 = matmul(self.J_HxT, M1)  # Matrix operation
        output = matmul(Gamma_predict, M2)  # Matrix operation
        return output

    def compute_Gamma_update(self, Gamma_predict, Kalman_gain):
        M1 = matmul(self.J_Hx, Gamma_predict)  # Matrix operation
        M2 = matmul(Kalman_gain, M1)  # Matrix operation
        output = subtract(Gamma_predict, M2)  # Matrix operation
        return output

    # Predition step: x_{k|k-1}, Gamma_{k|k-1| = predict(x_{k-1|k-1}, Gamma_{k-1|k-1}, t, k)
    def predict(self, x, Gamma, t, k):
        x_predict = self.F(x, t) + self.additive_evolution_vector + self.mean_epsilon[:, k + 1]
        Gamma_predict = self.compute_Gamma_predict(x, t, Gamma) + self.Gamma_epsilon[k + 1]
        return x_predict, Gamma_predict

    # x_{k|k}, Gamma_{k|k} = update(x_{k-1|k-1}, Gamma_{k-1|k-1}, y_k, t, k)
    def update(self, x_predict, Gamma_predict, y, t, k):
        # Innovation steps:
        innovation = y - self.H(x_predict, t) - self.mean_nu[:, k + 1]
        Gamma_innovation = self.compute_Gamma_innovation(x_predict, t, k, Gamma_predict) + self.Gamma_nu[k + 1]
        # Kalman gain computation:
        Kalman_gain = self.compute_Kalman_gain(Gamma_predict, Gamma_innovation)
        # Update steps:
        x_update = x_predict + matmul(Kalman_gain, innovation)
        Gamma_update = self.compute_Gamma_update(Gamma_predict, Kalman_gain)
        return x_update, Gamma_update


#######################################################
# Fixed-interval extended Kalman smoother function:
def compute_fixed_interval_extended_Kalman_smoother(J_F, dt, NT, N, x, Gamma, x_predict, Gamma_predict):
    t = dt * (NT - 1)  # Initialising time
    x_smoothed = zeros([N, NT])  # Initialising smoothed state
    Gamma_smoothed = zeros([NT, N, N])  # Initialising smoothed covariance
    x_smoothed[:, NT - 1] = x[:, NT - 1]  # Adding final filered state to smoothed state
    Gamma_smoothed[NT - 1] = Gamma[NT - 1]  # Adding final filtered covariance to smoothed covariance
    for k in tqdm(reversed(range(NT - 1))):  # Iterating over time backwards
        # Evolution model Jacobians:
        J_FxT = transpose(J_F(x[:, k], t))  # Transpose of evolution Jacobian evaluation at x_{t-1,t-1}
        # Computing matrix A:
        M1 = inv(Gamma_predict[k + 1])  # Matrix operation
        M2 = matmul(J_FxT, M1)  # Matrix operation
        A = matmul(Gamma[k], M2)  # Matrix A computation
        # Computing estimate difference:
        x_diff = subtract(x_smoothed[:, k + 1], x_predict[:, k + 1])
        Gamma_diff = subtract(Gamma_smoothed[k + 1], Gamma_predict[k + 1])
        # Computing smoothed estimate:
        x_smoothed[:, k] = x[:, k] + matmul(A, x_diff)
        # Computing smoothed covariance:
        AT = transpose(A)
        M2 = matmul(Gamma_diff, AT)
        Gamma_smoothed[k] = Gamma[k] + matmul(A, M2)
        # Next time step:
        t = (k - 1) * dt
    return x_smoothed, Gamma_smoothed
