from ops import sum_double_op, matmul_v1_op
import torch
import time
import numpy as np

class Timer:
    def __init__(self, op_name):
        self.begin_time = 0
        self.end_time = 0
        self.op_name = op_name

    def __enter__(self):
        torch.cuda.synchronize()
        self.begin_time = time.time()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        self.end_time = time.time()
        print(f"Average time cost of {self.op_name} is {(self.end_time - self.begin_time) * 1000:.4f} ms")

if __name__ == '__main__':
    n = 10
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    array1 = torch.ones(n, dtype=torch.float32, device=device, requires_grad=True)
    array2 = torch.ones(n, dtype=torch.float32, device=device, requires_grad=True)
    with Timer("sum_double"):
        ans = sum_double_op(array1, array2)
    assert (ans == 2).all()

    k = 10
    m = 10
    # mat1 = torch.tensor(np.random.rand(m, k), dtype=torch.float32, device=device, requires_grad=True)
    # mat2 = torch.tensor(np.random.rand(k, n), dtype=torch.float32, device=device, requires_grad=True)
    mat1 = torch.tensor(np.ones((m, k)), dtype=torch.float32, device=device, requires_grad=True)
    mat2 = torch.tensor(np.ones((k, n)), dtype=torch.float32, device=device, requires_grad=True)    
    with Timer("matmu_v1"):
        ans = matmul_v1_op(mat1, mat2)
    torch_ans = torch.matmul(mat1, mat2)
    assert ans.shape == torch_ans.shape, "matmul shape not equal."
    assert torch.equal(torch_ans, ans), "matmul value not equal."
