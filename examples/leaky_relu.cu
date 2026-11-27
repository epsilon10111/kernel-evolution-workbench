#include <cuda_runtime.h>
#include <cstddef>

__global__ void leaky_relu(const float* input, float alpha, float* output, size_t size) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size) {
        const float value = input[index];
        output[index] = value > 0.0f ? value : alpha * value;
    }
}

extern "C" void solution(const float* input, float alpha, float* output, size_t n, size_t m) {
    const size_t size = n * m;
    constexpr unsigned block = 256;
    leaky_relu<<<(size + block - 1) / block, block>>>(input, alpha, output, size);
}
