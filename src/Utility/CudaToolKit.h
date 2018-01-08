#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include "cutil_math.h"

/// On Cuda device with capabilities >= 2.0(prior to fermi) the architecture is 32 bit,
/// 24bit multiplication may be slower as it should simulate 24bit.
/// It may be dropped later
inline __device__ uint2 thr2pos2() {
#ifdef __CUDACC__
    return make_uint2( __umul24(blockDim.x, blockIdx.x) + threadIdx.x,
                        __umul24(blockDim.y, blockIdx.y) + threadIdx.y);
#else
    return make_uint2(0);
#endif
}

#ifndef CUDASafeCall
#define CUDASafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)

namespace {
inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
    if(cudaSuccess != err) {
        std::cout << file << "(" << line << ")" << " : CudaSafeCall() Runtime API error : "
            << cudaGetErrorString(err) << "." << std::endl;
        exit(-1);
    }
}
}

#endif

// This structure can reference specified data only.
struct MemoryReference {
    MemoryReference(void *data = NULL): m_data(data) {
    }
    void *m_data;
};

struct HostAllocator {
    HostAllocator(): m_data(NULL) {
    }
    ~HostAllocator() {
        cudaFreeHost(m_data);
    }

    void Allocate(uint size) {
        cudaHostAlloc(&m_data, size, cudaHostAllocDefault);
    }
    void *m_data;
};

struct DeviceAllocator {
    DeviceAllocator(): m_data(NULL) {
    }
    ~DeviceAllocator() {
        cudaFree(m_data);
    }

    void Allocate(uint size) {
        cudaMalloc(&m_data, size);
    }
    void *m_data;
};

struct HostDeviceAllocator {
    HostDeviceAllocator(): m_data(NULL) {
    }
    ~HostDeviceAllocator() {
        cudaFreeHost(m_data);
    }

    void Allocate(uint size) {
        cudaHostAlloc(&m_data, size, cudaHostAllocMapped);
    }
    void *GetDevicePtr() const {
        void *device_ptr;
        cudaHostGetDevicePointer(&device_ptr, m_data, 0);
        return device_ptr;
    }
    void *m_data;
};

inline int divup(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

inline dim3 divup(uint2 a,dim3 b) {
    return dim3(divup(a.x, b.x), divup(a.y, b.y));
}

inline dim3 divup(dim3 a, dim3 b) {
    return dim3(divup(a.x, b.x), divup(a.y, b.y), divup(a.z, b.z));
}

template <typename OTHER>
inline void image_copy( MemoryReference & to, const OTHER & from, uint size ){
    to.m_data = from.m_data;
}

inline void image_copy( HostAllocator& to, const HostAllocator& from, uint size ){
    cudaMemcpy(to.m_data, from.m_data, size, cudaMemcpyHostToHost);
}

inline void image_copy( HostAllocator& to, const DeviceAllocator& from, uint size ){
    cudaMemcpy(to.m_data, from.m_data, size, cudaMemcpyDeviceToHost);
}

inline void image_copy( HostAllocator& to, const HostDeviceAllocator& from, uint size ){
    cudaMemcpy(to.m_data, from.m_data, size, cudaMemcpyHostToHost);
}

inline void image_copy( DeviceAllocator& to, const MemoryReference & from, uint size ){
    cudaMemcpy(to.m_data, from.m_data, size, cudaMemcpyDeviceToDevice);
}

inline void image_copy( DeviceAllocator& to, const HostAllocator& from, uint size ){
    cudaMemcpy(to.m_data, from.m_data, size, cudaMemcpyHostToDevice);
}

inline void image_copy( DeviceAllocator& to, const DeviceAllocator& from, uint size ){
    cudaMemcpy(to.m_data, from.m_data, size, cudaMemcpyDeviceToDevice);
}

inline void image_copy( DeviceAllocator& to, const HostDeviceAllocator& from, uint size ){
    cudaMemcpy(to.m_data, from.GetDevicePtr(), size, cudaMemcpyDeviceToDevice);
}

inline void image_copy( HostDeviceAllocator& to, const HostAllocator& from, uint size ){
    cudaMemcpy(to.m_data, from.m_data, size, cudaMemcpyHostToHost);
}

inline void image_copy( HostDeviceAllocator& to, const DeviceAllocator& from, uint size ){
    cudaMemcpy(to.GetDevicePtr(), from.m_data, size, cudaMemcpyDeviceToDevice);
}

inline void image_copy( HostDeviceAllocator& to, const HostDeviceAllocator& from, uint size ){
    cudaMemcpy(to.m_data, from.m_data, size, cudaMemcpyHostToHost);
}

template <typename T, typename Allocator = MemoryReference>
struct Image: public Allocator {
    typedef T PixelType;
    uint2 m_size;

    Image(): Allocator(), m_size(make_uint2(0)) {
    }
    Image(const uint2 &s) {
        Allocate(s);
    }

    void Allocate(const uint2 &s) {
        if(s.x == m_size.x && s.y == m_size.y) {
            return;
        }
        Allocator::Allocate(s.x * s.y * sizeof(PixelType));
        m_size = s;
    }

    uint2 Size() {
        return m_size;
    }

    __device__ PixelType & el() {
        return operator[](thr2pos2());
    }

    __device__ const PixelType & el() const {
        return operator[](thr2pos2());
    }

    __device__ PixelType & operator[](const uint2 pos) {
        return static_cast<PixelType *>(Allocator::m_data)[pos.x + m_size.x * pos.y];
    }

    __device__ const PixelType & operator[](const uint2 pos) const {
        return static_cast<const PixelType *>(Allocator::m_data)[pos.x + m_size.x * pos.y];
    }

    operator Image<PixelType>() {
        return Image<PixelType>(m_size, Allocator::m_data);
    }

    template<typename A>
        Image<T, Allocator> & operator=(const Image<T, A> & other) {
            image_copy(*this, other, other.m_size.x * other.m_size.y * sizeof(T));
            m_size = other.m_size; //!TODO() should be changed here
            return *this;
        }

    Image<PixelType> GetDeviceImage() {
        return Image<PixelType>(m_size, Allocator::GetDevicePtr());
    }

    PixelType *Data() {
        return static_cast<PixelType *>(Allocator::m_data);
    }

    const PixelType *Data() const {
        return static_cast<const PixelType *>(Allocator::m_data);
    }

};

// Reference only Image template
template <typename T>
struct Image<T, MemoryReference>: public MemoryReference {
    typedef T PixelType;
    uint2 m_size;

    Image():m_size(make_uint2(0, 0)) {
    }
    Image(const uint2 &s, void *d):MemoryReference(d), m_size(s) {
    }

    uint2 Size() {
        return m_size;
    }

    __device__ PixelType & el() {
        return operator[](thr2pos2());
    }

    __device__ const PixelType & el() const {
        return operator[](thr2pos2());
    }

    __device__ PixelType &operator[](const uint2 &pos) {
        return static_cast<PixelType *>(MemoryReference::m_data)[pos.x + m_size.x * pos.y];
    }

    __device__ const PixelType &operator[](const uint2 &pos) const {
        return static_cast<PixelType *>(MemoryReference::m_data)[pos.x + m_size.x * pos.y];
    }

    PixelType *Data() {
        return static_cast<PixelType *>(MemoryReference::m_data);
    }

    const PixelType *Data() const {
        return static_cast<const PixelType *>(MemoryReference::m_data);
    }
};

struct TrackData {
    int result;
    float error;
    float J[6];
};

