#include <torch/extension.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;

void cuda_forward(int B, int T, int C, int H, bf16 *q, bf16 *k, bf16 *v, float *a, bf16 *S, bf16 *Z);
void cuda_backward(int B, int T, int C, int H, bf16 *q, bf16 *k, bf16 *v, float *a, float *aa, bf16 *gS, bf16 *gZ, bf16 *gr, bf16 *gk, bf16 *gv, bf16 *ga);

void forward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &S, torch::Tensor &Z) {
    cuda_forward(B, T, C, H, q.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), a.data_ptr<float>(), S.data_ptr<bf16>(), Z.data_ptr<bf16>());
}
void backward(int64_t B, int64_t T, int64_t C, int64_t H, torch::Tensor &q, torch::Tensor &k, torch::Tensor &v, torch::Tensor &a, torch::Tensor &aa, torch::Tensor &gS, torch::Tensor &gZ, torch::Tensor &gr, torch::Tensor &gk, torch::Tensor &gv, torch::Tensor &ga) {
    cuda_backward(B, T, C, H, q.data_ptr<bf16>(), k.data_ptr<bf16>(), v.data_ptr<bf16>(), a.data_ptr<float>(), aa.data_ptr<float>(), gS.data_ptr<bf16>(), gZ.data_ptr<bf16>(), gq.data_ptr<bf16>(), gk.data_ptr<bf16>(), gv.data_ptr<bf16>(), ga.data_ptr<bf16>());
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "LNSSM forward");
    m.def("backward", &backward, "LNSSM backward");
}

TORCH_LIBRARY(LNSSM, m) {
    m.def("forward", forward);
    m.def("backward", backward);
}