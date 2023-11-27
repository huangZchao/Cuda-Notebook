# import torch
# from torch.autograd import Function
# import matmul_v1

# class MatmulV1(Function):
#     @staticmethod
#     def forward(ctx, x1, x2):
#         """sum_double function forward.
#         Args:
#             x1 (torch.Tensor): [m, n]
#             x2 (torch.Tensor): [m, n]
        
#         Returns:
#             y (torch.Tensor): [m, n]
#         """
#         x1 = x1.float()
#         x2 = x2.float()
#         m, k = x1.shape
#         k, n = x2.shape

#         y = x1.new_zeros([m, n])

#         matmul_v1.matmul_v1_gpu(x1.contiguous(), x2.contiguous(), y, m, n, k)
        
#         ctx.save_for_backward(x1, x2)

#         return y
    
#     @staticmethod
#     def backward(ctx, grad_out):
#         """
#         Args:
#             grad_out: gradient of previous layer 
#         Returns:
#             grad_x1: gradient of x1
#             grad_x2: gradient of x2
#         """
#         x1, x2  = ctx.saved_tensors
#         grad_x1 = grad_out * x2
#         grad_x2 = grad_out * x1
#         return grad_x1, grad_x2
    

# matmul_v1_op = MatmulV1.apply
matmul_v1_op = 1