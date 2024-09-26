import torch
import torch.nn as nn
from torch.autograd import Function

from ..compression.hosvd import hosvd, restore_hosvd
###### HOSVD base on variance #############
class Linear_HOSVD_op(Function):
    @staticmethod
    def forward(ctx, *args):
        input, weight, bias, var, k_hosvd = args

        # Infer output
        output = torch.matmul(input, weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        if input.dim() == 2:
            # Perform decomposition
            S, [u0, u1] = hosvd(input, var=var)
            
            # Log information for estimating activation memory
            k_hosvd[0].append(u0.shape[1])
            k_hosvd[1].append(u1.shape[1])
            k_hosvd[2].append(1) # Choose 1 because insert 0 in raw_shape, so it's ok
            k_hosvd[3].append(1)
            k_hosvd[4].append(torch.tensor([input.shape[0], input.shape[1], 0, 0]))

            # Save information for backpropagation
            ctx.save_for_backward(S, u0, u1, weight, bias)
        else:
            # Perform decomposition
            S, [u0, u1, u2, u3] = hosvd(input, var=var)

            # Log information for estimating activation memory
            k_hosvd[0].append(u0.shape[1])
            k_hosvd[1].append(u1.shape[1])
            k_hosvd[2].append(u2.shape[1])
            k_hosvd[3].append(u3.shape[1])
            k_hosvd[4].append(input.shape)

            # Save information for backpropagation
            ctx.save_for_backward(S, u0, u1, u2, u3, weight, bias)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if len(ctx.saved_tensors) == 5:
            # Load the information that is saved from forwardpass
            S, U0, U1, weight, bias = ctx.saved_tensors
            input = restore_hosvd(S, [U0, U1])

        elif len(ctx.saved_tensors) == 7:
            # Load the information that is saved from forwardpass
            S, U0, U1, U2, U3, weight, bias = ctx.saved_tensors
            input = restore_hosvd(S, [U0, U1, U2, U3])
    
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            # grad_input = grad_output.mm(weight)
            grad_input = torch.matmul(grad_output, weight)

        if ctx.needs_input_grad[1]:
            if (grad_output.dim() == 4):
                grad_weight = torch.matmul(grad_output.permute(0, 1, 3, 2), input)
            elif (grad_output.dim() == 2):
                # grad_weight = grad_output.t().mm(input)
                grad_weight = torch.matmul(grad_output.t(), input)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias, None, None

class Linear_HOSVD(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
            activate=False,
            var=0.9,
            k_hosvd = None):
        super(Linear_HOSVD, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.activate = activate
        self.var = var
        self.k_hosvd = k_hosvd

    def forward(self, input):
        if self.activate and torch.is_grad_enabled(): # Training mode
            output = Linear_HOSVD_op.apply(input, self.weight, self.bias, self.var, self.k_hosvd)
        else: # activate is False or Validation mode
            output = super().forward(input)
        return output
    

def wrap_linearHOSVD(linear, SVD_var, active, k_hosvd):
    has_bias = (linear.bias is not None)
    new_linear = Linear_HOSVD(in_features=linear.in_features,
                        out_features=linear.out_features,
                        bias=has_bias,
                        activate=active,
                        var=SVD_var,
                        k_hosvd = k_hosvd
                        )
    new_linear.weight.data = linear.weight.data
    if new_linear.bias is not None:
        new_linear.bias.data = linear.bias.data
    return new_linear