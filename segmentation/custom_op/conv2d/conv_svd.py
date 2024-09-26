import torch as th
from torch.autograd import Function
from typing import Any
from torch.nn.functional import conv2d
import torch.nn as nn
from ..compression.hosvd import unfolding, truncated_svd


###### SVD by choosing principle components based on variance
def restore_tensor(Uk_Sk, Vk_t, shape):
    reconstructed_matrix = th.matmul(Uk_Sk, Vk_t)
    shape = tuple(shape)
    return reconstructed_matrix.view(shape)

###############################################################
class Conv2d_SVD_op(Function):
    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        input, weight, bias, stride, dilation, padding, groups, var, svd_size = args

        # Perform convolution
        output = conv2d(input, weight, bias, stride, padding, dilation=dilation, groups=groups)

        # Perform decomposition
        input_Uk, input_Sk, input_Vk_t = truncated_svd(unfolding(0, input), var=var) # Apply SVD compression on the 1st mode of input
        input_Uk_Sk = th.matmul(input_Uk, th.diag_embed(input_Sk))
        # Log information for estimating activation memory
        if svd_size is not None:
            svd_size.append(th.tensor([input_Uk_Sk.shape[0], input_Vk_t.shape[0], input_Vk_t.shape[1]], device=input_Uk_Sk.device))

        # Save tensors for backward pass
        ctx.save_for_backward(input_Uk_Sk, input_Vk_t, th.tensor(input.shape), weight, bias)
        ctx.stride = stride
        ctx.padding = padding 
        ctx.dilation = dilation
        ctx.groups = groups

        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        # Retrieve saved tensors
        input_Uk_Sk, input_Vk_t, input_shape, weight, bias = ctx.saved_tensors
        input = restore_tensor(input_Uk_Sk, input_Vk_t, input_shape)
        stride = ctx.stride
        padding = ctx.padding 
        dilation = ctx.dilation
        groups = ctx.groups
        grad_input = grad_weight = grad_bias = None
        grad_output, = grad_outputs

        # Compute gradient with respect to the input
        if ctx.needs_input_grad[0]:
            grad_input = nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)
        
        # Compute gradient with respect to the weights
        if ctx.needs_input_grad[1]:
            grad_weight = nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0,2,3)).squeeze(0)
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None # Gradients corrs to args in forward function
    
class Conv2d_SVD(nn.Conv2d):
    """
    Custom Conv2D layer with SVD-based decomposition.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            padding=0,
            device=None,
            dtype=None,
            activate=False,
            var=1,
            svd_size=None
    ) -> None:
        if kernel_size is int:
            kernel_size = [kernel_size, kernel_size]
        if padding is int:
            padding = [padding, padding]
        if dilation is int:
            dilation = [dilation, dilation]
        super(Conv2d_SVD, self).__init__(in_channels=in_channels,
                                        out_channels=out_channels,
                                        kernel_size=kernel_size,
                                        stride=stride,
                                        dilation=dilation,
                                        groups=groups,
                                        bias=bias,
                                        padding=padding,
                                        padding_mode='zeros',
                                        device=device,
                                        dtype=dtype)
        self.activate = activate
        self.var = var
        self.svd_size = svd_size

    def forward(self, x: th.Tensor) -> th.Tensor:
        if self.activate and th.is_grad_enabled(): # Training mode
            y = Conv2d_SVD_op.apply(x, self.weight, self.bias, self.stride, self.dilation, self.padding, self.groups, self.var, self.svd_size)
        else: # activate is False or Validation mode
            y = super().forward(x)
        return y

def wrap_convSVD(conv, SVD_var, active, svd_size = None):
    """
    Wrap an existing Conv2D layer into a Conv2d_SVD layer to apply SVD compression.

    Args:
        conv (nn.Conv2d): The original Conv2D layer to wrap.
        SVD_var (float): The explained variance threshold for SVD.
        active (bool): Whether to activate SVD.
        k_hosvd (list): A list to log the size of activation memory during training.
    
    Returns:
        Conv2d_SVD: The wrapped Conv2D layer with SVD compression.
    """

    new_conv = Conv2d_SVD(in_channels=conv.in_channels,
                         out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         stride=conv.stride,
                         dilation=conv.dilation,
                         bias=conv.bias is not None,
                         groups=conv.groups,
                         padding=conv.padding,
                         activate=active,
                         var=SVD_var,
                         svd_size=svd_size
                         )
    new_conv.weight.data = conv.weight.data
    if new_conv.bias is not None:
        new_conv.bias.data = conv.bias.data
    return new_conv