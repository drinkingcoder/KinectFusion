#include "Volume.h"

__global__ void SetVolumeKernel(Volume volume, const float2 val) {
    uint3 pos = make_uint3(thr2pos2());
    for (pos.z = 0; pos.z < volume.m_size.z; ++pos.z) {
        volume.set(pos, val);
    }
}

__forceinline__ __device__ bool InRange(float x, float a, float b) {
    return a <= x && x <= b;
}

__global__ void SetBoxKernel(Volume volume, const float3 min_corner, const float3 max_corner, const float val) {
    uint3 pos = make_uint3(thr2pos2());
    for (pos.z = 0; pos.z < volume.m_size.z; pos.z++) {
        const float3 p = volume.pos(pos);
        if ( InRange(p.x, min_corner.x, max_corner.x) 
                && InRange(p.y, min_corner.y, max_corner.y) 
                && InRange(p.z, min_corner.z, max_corner.z) ) {
            volume.set(pos, make_float2(val, 0.0f));
        }
    }
}

__global__ void PrintInnerVoxelsKernel(Volume volume) {
    uint3 pos = make_uint3(thr2pos2());
    for (pos.z = 0; pos.z < volume.m_size.z; pos.z++) {
        if (volume[pos].x < 0) {
            printf("(%d, %d, %d)\n",pos.x, pos.y, pos.z);
        }
    }
}

void Volume::PrintInnerVoxels() {
    dim3 block3(32, 16);
    PrintInnerVoxelsKernel<<<divup(dim3(m_size.x, m_size.y), block3), block3>>>(*this);
}

void Volume::SetVolumeWrap(const float val) {
    dim3 block3(32, 16);
    SetVolumeKernel<<<divup(dim3(m_size.x, m_size.y), block3), block3>>>(*this, make_float2(val, 0.0f));
}

void Volume::SetBoxWrap(const float3 min_corner, const float3 max_corner, const float val) {
    dim3 block3(32, 16);
    std::cout << "SetBoxWrap: " << std::endl;
    SetBoxKernel<<<divup(dim3(m_size.x, m_size.y), block3), block3>>>(*this,
            min_corner,
            max_corner,
            val);
}

void Volume::Reset() {
    dim3 block(32, 16);
    dim3 grid = divup(dim3(m_size.x, m_size.y), block);
    SetVolumeKernel<<<grid, block>>>(*this, make_float2(1.0f, 0.0f));
}

