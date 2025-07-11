#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               const F *__restrict__ const _q, const F *__restrict__ const _k, const F *__restrict__ const _v, const float *__restrict__ _a,
                               F *__restrict__ const _S, F *__restrict__ const _Z)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _a += h*_N_;


    __shared__ float q[_N_], k[_N_], a[_N_];
    float state_S[_N_] = {0};
    float stateZ = 0;
    
    __syncthreads();
    a[i] = _a[i];
    __syncthreads();

    for (int t = b*T*C + h*_N_ + i; t < (b+1)*T*C + h*_N_ + i; t += C)
    {
        __syncthreads();
        q[i] = float(_q[t]);
        k[i] = float(_k[t]);
        __syncthreads();

        const float v = float(_v[t]);
        float S = 0;
        
        stateZ = stateZ * a[i] + k[i];
        _Z[t] = F(stateZ);

        #pragma unroll
        for (int j = 0; j < _N_; j+=4)
        {
            const float4& q_ = (float4&)(q[j]);
            const float4& k_ = (float4&)(k[j]);
            const float4& a_ = (float4&)(a[j]);
            float4& s = (float4&)(state_S[j]);
            float4 x;

            x.x = k_.x * v;
            x.y = k_.y * v;
            x.z = k_.z * v;
            x.w = k_.w * v;

            s.x = s.x * a_.x + x.x;
            s.y = s.y * a_.y + x.y;
            s.z = s.z * a_.z + x.z;
            s.w = s.w * a_.w + x.w;

            S += q_.x * s.x;
            S += q_.y * s.y;
            S += q_.z * s.z;
            S += q_.w * s.w;


        }
        _S[t] = F(S);
    }
}

template <typename F>
__global__ void kernel_backward(const int B, const int T, const int C, const int H,
    const F *__restrict__ const _q, const F *__restrict__ const _k, const F *__restrict__ const _v, const float *__restrict__ _a, const float *__restrict__ __a, const F *__restrict__ const _gS, const F *__restrict__ const _gZ,
    F *__restrict__ const _gq, F *__restrict__ const _gk, F *__restrict__ const _gv, F *__restrict__ const _ga)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _a += h*_N_;
    __a += h*_N_;

    __shared__ float a_[_N_];
    __shared__ float q[_N_], k[_N_], v[_N_], gS[_N_];
    __syncthreads();
    a_[i] = _a[i];
    __syncthreads();

    const float a = a_[i];
    const float aa = __a[i];


    float state[_N_] = {0}, saaaa[_N_] = {0}, sbbbb[_N_] = {0}, scccc[_N_] = {0}, sdddd[_N_] = {0};

    float ga = 0;
    const int t000 = b*T*C + h*_N_ + i;
    const int t111 = (b+1)*T*C + h*_N_ + i;
    const int t222 = t111 - C;

    for (int t = t000; t < t111; t += C)
    {
        __syncthreads();
        v[i] = float(_v[t]);
        gS[i] = float(_gS[t]);
        __syncthreads();

        const float k = float(_k[t]);
        float gq = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = state[j];
            float x = k * v[j];
            s = s * a + x;
            gq += s * gS[j];
            
        }
        _gq[t] = F(gq);

    }

    float z = 0;
    float z2 = 0;    
    for (int t = t000; t < t222; t += C)
    {
        __syncthreads();
        v[i] = float(_v[t]);
        gS[i] = float(_gS[t + C]);
        __syncthreads();

        const float k = float(_k[t]);
        float ga_s = 0;

        const float gZ = float(_gZ[t]);       
 
        float tmp = a * (k + z);
        z = tmp;
        z2 = tmp + a * z2;
        ga += z2 * gZ;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = saaaa[j];
            float& s2 = sbbbb[j];
            float x = k * v[j];
            
            float tmp = a * (x + s);
            s = tmp;
            s2 = tmp + a * s2;
            ga_s += s2 * gS[j];
        }
        ga += float(_q[t + C]) * ga_s;
    }    
    _ga[b*C + h*_N_ + i] = F(aa * ga);

    float z3 = 0;
    for (int t = t111 - C; t >= t000; t -= C)
    {
        __syncthreads();
        v[i] = float(_v[t]);
        gS[i] = float(_gS[t]);
        __syncthreads();

        const float qq = float(_q[t]);
        float gk = 0;
        
        const float gZ = float(_gZ[t]);
        z3 = gZ + z3 * a;
        gk += z3;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = scccc[j];
            float x = qq * gS[j];
            s = x + s * a;
            gk += s * v[j];
            
        }
        _gk[t] = F(gk);
    }

    for (int t = t111 - C; t >= t000; t -= C)
    {
        __syncthreads();
        q[i] = float(_q[t]);
        k[i] = float(_k[t]);
        __syncthreads();

        const float gSS = float(_gS[t]);
        float gv = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = sdddd[j];
            float x = gSS * q[j];
            s = x + s * a_[j];
            gv += s * k[j];
            
        }
        _gv[t] = F(gv);
    }
}

void cuda_forward(int B, int T, int C, int H, bf16 *q, bf16 *k, bf16 *v, float *a, bf16 *S, bf16 *Z)
{
    assert(H*_N_ == C);
    assert(_N_%4 == 0);
    kernel_forward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, q, k, v, a, S, Z);
}

void cuda_backward(int B, int T, int C, int H, bf16 *q, bf16 *k, bf16 *v, float *a, float *aa, bf16 *gS, bf16 *gZ, bf16 *gq, bf16 *gk, bf16 *gv, bf16 *ga)
{
    assert(H*_N_ == C);
    assert(_N_%4 == 0);
    kernel_backward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, q, k, v, a, aa, gS, gZ, gq, gk, gv, ga);
}