"""
Compilation functions for the GDE evolution operator
"""


#######################################################
# Local modules:
from basic_tools import euler, rk4


#######################################################
# Gets function f such that dx(t) / dt = f(x(t), t) depending on unknown processes:
def get_f(model):
    if 'condensation' in model.unknowns and 'deposition' in model.unknowns and 'source' in model.unknowns:
        def f(x, _):
            alpha = x[0: model.N]
            gamma = x[model.N: model.N + model.N_gamma]
            eta = x[model.N + model.N_gamma: model.N + model.N_gamma + model.N_eta]
            J = x[model.N + model.N_gamma + model.N_eta]
            return model.f_cond(alpha, gamma) + model.f_coag(alpha) + model.f_sorc(J) + model.f_depo(alpha, eta)
    elif 'condensation' in model.unknowns:
        def f(x, t):
            alpha = x[0: model.N]
            gamma = x[model.N: model.N + model.N_gamma]
            return model.f_cond(alpha, gamma) + model.f_coag(alpha) + model.f_sorc(t) + model.f_depo(alpha)
    elif 'deposition' in model.unknowns:
        def f(x, t):
            alpha = x[0: model.N]
            eta = x[model.N: model.N + model.N_eta]
            return model.f_cond(alpha) + model.f_coag(alpha) + model.f_sorc(t) + model.f_depo(alpha, eta)
    elif 'source' in model.unknowns:
        def f(x, _):
            alpha = x[0: model.N]
            J = x[model.N]
            return model.f_cond(alpha) + model.f_coag(alpha) + model.f_sorc(J) + model.f_depo(alpha)
    else:
        def f(alpha, t):
            return model.f_cond(alpha) + model.f_coag(alpha) + model.f_sorc(t) + model.f_depo(alpha)
    return f


#######################################################
# Gets function next_step such that x_{t + 1} = next_step(x_t, t) depending on time integrator:
def get_next_step(model):
    if model.time_integrator == 'euler':
        def next_step(x, t):
            alpha = x[0:model.N]
            return euler(model.f, alpha, model.dt, t, f_args=x)
    elif model.time_integrator == 'rk4':
        def next_step(x, t):
            return rk4(model.f, x, model.dt, t)
    else:
        def next_step(x, t):
            return model.f(x, t)
    return next_step
