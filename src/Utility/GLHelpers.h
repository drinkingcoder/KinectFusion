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
