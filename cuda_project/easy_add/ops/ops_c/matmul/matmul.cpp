#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.type().is_cuda(), #x, " must be a cuda tensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)


void matmul_v1_launcher(const float* a, const float* b, float* c, int m, int n, int k);

void matmul_v1_gpu(at::Tensor a_tensor, at::Tensor b_tensor, at::Tensor c_tensor, at::kInt m, at::kInt n, at::kInt k) {
    CHECK_INPUT(a_tensor);
    CHECK_INPUT(b_tensor);
    CHECK_INPUT(c_tensor);

    const float* a = a_tensor.data_ptr<float>();
    const float* b = b_tensor.data_ptr<float>();
    float* c = c_tensor.data_ptr<float>();
    
    matmul_v1_launcher(a, b, c, m, n, k);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_v1_gpu", &matmul_v1_gpu, "sum two arrays (CUDA)");
}