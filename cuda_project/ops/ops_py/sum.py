import torch
from torch.autograd import Function
import sum_double

class SumDouble(Function):
    @staticmethod
    def forward(ctx, x1, x2):
        """sum_double function forward.
        Args:
            x1 (torch.Tensor): [n,]
            x2 (torch.Tensor): [n,]
        
        Returns:
            y (torch.Tensor): [n,]
        """
        x1 = x1.float()
        x2 = x2.float()

        y = x1.new_zeros(x1.shape)

        sum_double.forward(x1.contiguous(), x2.contiguous(), y)

        return y
    
    @staticmethod
    def backward(ctx, grad_out):
        """
        Args:
            grad_out: gradient of previous layer 
        Returns:
            grad_x1: gradient of x1
            grad_x2: gradient of x2
        """
        grad_x1 = grad_out.clone()
        grad_x2 = grad_out.clone()
        return grad_x1, grad_x2
    

sum_double_op = SumDouble.apply