import torch

def kl_divergence(mu_p, logs_p, mu_q, logs_q, mask):
    var_p = torch.exp(2. * logs_p)
    var_q = torch.exp(2. * logs_q)
    kl = logs_q - logs_p - 0.5
    kl += 0.5 * (var_p + (mu_p - mu_q)**2) * var_q
    kl = torch.sum(kl * mask)
    l = kl / torch.sum(mask)
    return l