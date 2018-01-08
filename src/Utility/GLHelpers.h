#pragma once

#include "CudaToolKit.h"
#include <cuda_gl_interop.h> // includes cuda_gl_interop.h


template <typename T> struct gl;

template<> struct gl<unsigned char> {
    static const int format=GL_LUMINANCE;
    static const int type  =GL_UNSIGNED_BYTE;
};

template<> struct gl<uchar3> {
    static const int format=GL_RGB;
    static const int type  =GL_UNSIGNED_BYTE;
};

template<> struct gl<uchar4> {
    static const int format=GL_RGBA;
    static const int type  =GL_UNSIGNED_BYTE;
};

 template<> struct gl<float> {
    static const int format=GL_LUMINANCE;
    static const int type  =GL_FLOAT;
};

 template<> struct gl<float3> {
    static const int format=GL_RGB;
    static const int type  =GL_FLOAT;
};

template <typename T, typename A>
inline void glDrawPixels( const Image<T, A> & i ){
    ::glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    ::glPixelStorei(GL_UNPACK_ROW_LENGTH, i.m_size.x);
    ::glDrawPixels(i.m_size.x, i.m_size.y, gl<T>::format, gl<T>::type, i.Data());
}

inline void glDrawPixelsScale( const Image<float, HostDeviceAllocator> & i ){
    ::glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    ::glPixelStorei(GL_UNPACK_ROW_LENGTH, i.m_size.x);
    auto data = new float[i.m_size.x * i.m_size.y];
    cudaDeviceSynchronize();
    memcpy(data, i.m_data, sizeof(float) * i.m_size.x * i.m_size.y);
    for (auto x = 0; x < i.m_size.x; x++) {
        for (auto y = 0; y < i.m_size.y; y++) {
            data[x + y * i.m_size.x] *= 2000;
        }
    }
    ::glDrawPixels(i.m_size.x, i.m_size.y, gl<float>::format, gl<float>::type, data);
    delete[] data;
}
