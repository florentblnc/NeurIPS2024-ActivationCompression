import torch as th
import torch.nn as nn
from torch.autograd import Function
from ..compression.hosvd import truncated_svd, unfolding
###### SVD by choosing principle components based on variance
def restore_tensor(Uk_Sk, Vk_t, shape):
    reconstructed_matrix = th.matmul(Uk_Sk, Vk_t)
    shape = tuple(shape)
    return reconstructed_matrix.view(shape)
#############################
class Linear_SVD_op(Function):
    @staticmethod
    def forward(ctx, *args):
        input, weight, bias, var, svd_size = args

        # Infer output
        output = th.matmul(input, weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        # Perform decomposition
        input_Uk, input_Sk, input_Vk_t = truncated_svd(unfolding(0, input), var=var) # Apply SVD compression on the 1st mode of input
        input_Uk_Sk = th.matmul(input_Uk, th.diag_embed(input_Sk))

        # Log information for estimating activation memory
        svd_size.append(th.tensor([input_Uk_Sk.shape[0], input_Vk_t.shape[0], input_Vk_t.shape[1]], device=input_Uk_Sk.device))

        # Save information for backpropagation
        ctx.save_for_backward(input_Uk_Sk, input_Vk_t, th.tensor(input.shape), weight, bias)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Load the information that is saved from forwardpass
        input_Uk_Sk, input_Vk_t, input_shape, weight, bias = ctx.saved_tensors
        input = restore_tensor(input_Uk_Sk, input_Vk_t, input_shape)
    
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = th.matmul(grad_output, weight)

        if ctx.needs_input_grad[1]:
            if (grad_output.dim() == 4):
                grad_weight = th.matmul(grad_output.permute(0, 1, 3, 2), input)
            elif (grad_output.dim() == 2):
                grad_weight = th.matmul(grad_output.t(), input)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, grad_bias, None, None

class Linear_SVD(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
            activate=False,
            var=0.9,
            svd_size=None):
        super(Linear_SVD, self).__init__(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            device=device,
            dtype=dtype
        )
        self.activate = activate
        self.var = var
        self.svd_size = svd_size

    def forward(self, input):
        if self.activate and th.is_grad_enabled(): # Training mode
            output = Linear_SVD_op.apply(input, self.weight, self.bias, self.var, self.svd_size)
        else: # activate is False or Validation mode
            output = super().forward(input)
        return output
    

def wrap_linearSVD(linear, SVD_var, active, svd_size):
    has_bias = (linear.bias is not None)
    new_linear = Linear_SVD(in_features=linear.in_features,
                        out_features=linear.out_features,
                        bias=has_bias,
                        activate=active,
                        var=SVD_var,
                        svd_size=svd_size
                        )
    new_linear.weight.data = linear.weight.data
    if new_linear.bias is not None:
        new_linear.bias.data = linear.bias.data
    return new_linear