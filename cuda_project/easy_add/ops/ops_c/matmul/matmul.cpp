#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void matmul_v1_launcher(const float* a, const float* b, float* c, int m, int k, int n);
void matmul_v2_launcher(float* a, float* b, float* c, const int m, const int k, const int n);

void matmul_v1_gpu(at::Tensor a_tensor, at::Tensor b_tensor, at::Tensor c_tensor, int m, int k, int n) {
    CHECK_INPUT(a_tensor);
    CHECK_INPUT(b_tensor);
    CHECK_INPUT(c_tensor);

    const float* a = a_tensor.data_ptr<float>();
    const float* b = b_tensor.data_ptr<float>();
    float* c = c_tensor.data_ptr<float>();
    
    matmul_v1_launcher(a, b, c, m, k, n);
}

void matmul_v2_gpu(at::Tensor a_tensor, at::Tensor b_tensor, at::Tensor c_tensor, const int m, const int k, const int n) {
    CHECK_INPUT(a_tensor);
    CHECK_INPUT(b_tensor);
    CHECK_INPUT(c_tensor);

    float* a = a_tensor.data_ptr<float>();
    float* b = b_tensor.data_ptr<float>();
    float* c = c_tensor.data_ptr<float>();
    
    matmul_v2_launcher(a, b, c, m, k, n);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("matmul_v1_gpu", &matmul_v1_gpu, "sum two arrays v1 (CUDA)");
  m.def("matmul_v2_gpu", &matmul_v2_gpu, "sum two arrays v2 (CUDA)");
}