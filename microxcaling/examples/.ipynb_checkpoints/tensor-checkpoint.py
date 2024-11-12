
from mx import add_mx_args, get_mx_specs
import torch
import torch.nn.functional as F
import numpy as np
import argparse
from mx.mx_ops import quantize_mx_op

if __name__ == '__main__':
    # Add config arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_size", default=128)
    parser.add_argument("--device", default='cuda')
#    parser.add_argument("--a_elem_format", default='fp4_e2m1')
#    parser.add_argument("--block_size", default=16)
    # Add MX arguments
    parser = add_mx_args(parser)
    args = parser.parse_args()
    args.a_elem_format = 'fp4_e2m1_asym'
    args.block_size = args.hidden_size
    args.custom_cuda = True
    args.scale_mode = 2

    # Process args to obtain mx_specs
    mx_specs = get_mx_specs(args)
    assert(mx_specs != None)

    # Run MLP
#    x = np.random.randn(1, args.hidden_size)
    x = np.linspace(-2.5, 16.5, args.hidden_size)
#    x[:,0] = x[:,1]/2
#    x[:,1] = x[:,1]*5
#    x = [[0.1666]]
    #x = [[0.46875]]
    #x = [[30.]]
    x = torch.tensor(x, dtype=torch.float32, device=args.device)
    #x[0] = x[0]/4

    qx = quantize_mx_op(
        x,
        mx_specs,
        elem_format=mx_specs['a_elem_format'],
        axes=[-1],
        round=mx_specs["round_mx_output"],
    )
    import pdb; pdb.set_trace()

    print("DONE!")
