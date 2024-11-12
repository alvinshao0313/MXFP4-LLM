import sys
sys.path.append('/root/MX-QLLM')
from mx import add_mx_args, get_mx_specs
import torch
import torch.nn.functional as F
import numpy as np
from mx.mx_ops import quantize_mx_op
from mx import MxSpecs

if __name__ == '__main__':
    class Quantizer(object):
        def __init__(self, format, block_size, scale_mode):
            self.format = format
            self.block_size = block_size
            self.scale_bits = 8
            self.scale_mode = scale_mode
            self.mx_specs = MxSpecs(
                scale_bits=self.scale_bits,
                a_elem_format=format,
                block_size=block_size,
                custom_cuda=True,
                scale_mode=scale_mode,
            )
        def quant(self, x):
            x = torch.tensor(x,dtype=torch.float32,device='cuda')
            qx = quantize_mx_op(
                x,
                self.mx_specs,
                elem_format=self.format,
                axes=[-1],
                round=self.mx_specs["round_mx_output"],
            )
            return qx

    block_size = -1
    quantizer0 = Quantizer('fp4_e2m1_asym',block_size,0)
    quantizer3 = Quantizer('fp4_e2m1_asym',block_size,3)
    quantizer152 = Quantizer('fp4_e2m1_asym',block_size,152)
    with torch.no_grad():
        x = np.linspace(-1.3, 22.3, 1024)
        x = torch.tensor(x, dtype=torch.float32, device='cuda')
        qx0 = quantizer0.quant(x.data)
        qx3 = quantizer3.quant(x.data)
        qx152 = quantizer152.quant(x.data)

        print((x-qx0).pow(2).mean().item())
        print((x-qx3).pow(2).mean().item())
        print((x-qx152).pow(2).mean().item())

    print("DONE!")
