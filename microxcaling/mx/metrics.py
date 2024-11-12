import torch

def sqnr(a,b):
    if (a-b).sum() == 0: # all input is same
        return torch.ones(1)*1e+5 # infinite
    else:
        s_power = 0
        qn_power = 0
        # torch implementation
        s_power = torch.sum(torch.pow(a,2))
        qn_power = torch.sum(torch.pow(a-b,2))
        sqnr = 10.0*torch.log10(s_power/qn_power)
        return sqnr

def mse(a,b):
    return (a-b).pow(2).mean()
