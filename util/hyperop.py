import torch as th
import torch.nn as nn


PROJ_EPS = 1e-5
EPS = 1e-15


def project_hyp_vec(x):
    # To make sure hyperbolic embeddings are inside the unit ball.
    norm = th.sum(x**2, dim=-1, keepdim=True)
    
    return x * (1. - PROJ_EPS) / th.clamp(norm, 1. - PROJ_EPS)


def asinh(x):
    return th.log(x + (x**2 + 1)**0.5)


def acosh(x):
    return th.log(x + (x**2 - 1)**0.5)


def atanh(x):
    return 0.5*th.log((1 + x)/(1 - x))


def poinc_dist(u, v):
    m = mob_add(-u, v) + EPS
    atanh_x = th.norm(m, dim=-1, keepdim=True)
    dist_poincare = 2.0 * atanh(atanh_x)
    return dist_poincare


def euclid_dist(u, v):
    return th.norm(u - v, dim=-1, keepdim=True)


def mob_add(u, v):
    v = v + EPS
    
    norm_uv = 2*th.sum(u*v, dim=-1, keepdim=True)
    norm_u = th.sum(u**2, dim=-1, keepdim=True)
    norm_v = th.sum(v**2, dim=-1, keepdim=True)
    
    denominator = 1 + norm_uv + norm_v * norm_u
    result = (1 + norm_uv + norm_v)/denominator*u + (1 - norm_u)/denominator*v
    
    return project_hyp_vec(result)


def mob_scalar_mul(r, v):
    v = v + EPS
    norm_v = th.norm(v, dim=-1, keepdim=True)
    nomin = th.tanh(r*atanh(norm_v))
    result = nomin/norm_v*v
    
    return project_hyp_vec(result)


def mob_mat_mul(M, x):
    x = project_hyp_vec(x)
    Mx = x.matmul(M)
    Mx_norm = th.norm(Mx + EPS, dim=-1, keepdim=True)
    x_norm = th.norm(x + EPS, dim=-1, keepdim=True)
    
    return project_hyp_vec(th.tanh(Mx_norm/x_norm*atanh(x_norm)) / Mx_norm * Mx)


def mob_mat_mul_d(M, x, d_ball):
    x = project_hyp_vec(x)
    Mx = x.view(x.shape[0], -1).matmul(M.view(M.shape[0]*d_ball,M.shape[0]*d_ball)).view(x.shape)
    Mx_norm = th.norm(Mx + EPS, dim=-1, keepdim=True)
    x_norm = th.norm(x + EPS, dim=-1, keepdim=True)
    
    return project_hyp_vec(th.tanh(Mx_norm/x_norm*atanh(x_norm)) / Mx_norm * Mx)


def lambda_x(x):
    return 2. / (1 - th.sum(x**2, dim=-1, keepdim=True))


def exp_map_x(x, v):
    v = v + EPS
    second_term = th.tanh(lambda_x(x) * th.norm(v) / 2) / th.norm(v) * v
    return mob_add(x, second_term)


def log_map_x(x, y):
    diff = mob_add(-x, y) + EPS
    return 2. / lambda_x(x) * atanh(th.norm(diff, dim=-1, keepdim=True)) /\
            th.norm(diff, dim=-1, keepdim=True) * diff


def exp_map_zero(v):
    v = v + EPS
    norm_v = th.norm(v, dim=-1, keepdim=True)
    result = th.tanh(norm_v) / norm_v * v
    
    return project_hyp_vec(result)


def log_map_zero(y):
    diff = project_hyp_vec(y + EPS)
    norm_diff = th.norm(diff, dim=-1, keepdim=True)
    return atanh(norm_diff)/norm_diff*diff


def mob_pointwise_prod(x, u):
    # x is hyperbolic, u is Euclidean
    x = project_hyp_vec(x + EPS)
    Mx = x * u
    Mx_norm = th.norm(Mx + EPS, dim=-1, keepdim=True)
    x_norm = th.norm(x, dim=-1, keepdim=True)

    result = th.tanh(Mx_norm / x_norm * atanh(x_norm)) / Mx_norm * Mx
    return project_hyp_vec(result)
