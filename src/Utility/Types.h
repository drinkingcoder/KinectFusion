#pragma once

#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <Eigen/Core>

#include <cmath>

#include "cutil_math.h"

//#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

typedef Eigen::Affine3f Affine3f;

typedef Eigen::Matrix<float, 4, 4> Matrix4f;
typedef Eigen::Matrix<float, 3, 3> Matrix3f;

typedef Eigen::Matrix<float, 4, 1> Vector4f;
typedef Eigen::Matrix<float, 3, 1> Vector3f;

typedef Eigen::AngleAxisf AngleAxisf;

/*
inline __host__ __device__ float3 operator*(const Affine3f & C, const float3 & p) {
    float3 res;
    Eigen::Map<Vector3f> res_map(&res);
    res_map = C.linear() * Eigen::Map<Vector3f>(&p) + C.translation();
    return res;
}
*/

inline __host__ __device__ float3 operator*(const Matrix3f & M, const float3 & p) {
    float3 res;
    Eigen::Map<Vector3f> res_map(reinterpret_cast<float*>(&res));
    res_map = M * Eigen::Map<const Vector3f>(reinterpret_cast<const float*>(&p));
    return res;
}

inline __host__ __device__ float3 rotate(const Matrix4f & M, const float3 & p) {
    float3 res;
    Eigen::Map<Vector3f> res_map(reinterpret_cast<float*>(&res));
    res_map = M.block(0, 0, 3, 3) * Eigen::Map<const Vector3f>(reinterpret_cast<const float*>(&p));
    return res;
}

inline __host__ __device__ float3 transform(const Matrix4f & M, const float3 & p) {
    float3 res;
    Eigen::Map<Vector3f> res_map(reinterpret_cast<float*>(&res));
    res_map = M.block(0, 0, 3, 3) * Eigen::Map<const Vector3f>(reinterpret_cast<const float*>(&p)) + M.block(0, 3, 3,  1);
    return res;
}

/*
struct Matrix4 {
    float4 data[4];
    inline __host__ __device__ Matrix4() {
        memset(data, 0 , 4 * 4 * sizeof(float));
        data[0].x = data[1].y = data[2].z = data[3].w = 1.0f;
    }
    inline __host__ __device__ float3 get_translation() const {
        return make_float3(data[0].w, data[1].w, data[2].w);
    }
};

std::ostream & operator<<( std::ostream & out, const Matrix4 & m );
Matrix4 operator*( const Matrix4 & A, const Matrix4 & B);
Matrix4 inverse( const Matrix4 & A );

inline __host__ __device__ float4 operator*( const Matrix4 & M, const float4 & v){
    return make_float4( dot(M.data[0], v), dot(M.data[1], v), dot(M.data[2], v), dot(M.data[3], v));
}

inline __host__ __device__ float3 operator*( const Matrix4 & M, const float3 & v){
    return make_float3(
        dot(make_float3(M.data[0]), v) + M.data[0].w,
        dot(make_float3(M.data[1]), v) + M.data[1].w,
        dot(make_float3(M.data[2]), v) + M.data[2].w);
}

inline __host__ __device__ float3 rotate( const Matrix4 & M, const float3 & v){
    return make_float3(
        dot(make_float3(M.data[0]), v),
        dot(make_float3(M.data[1]), v),
        dot(make_float3(M.data[2]), v));
}

inline Matrix4 getCameraMatrix( const float4 & k ){
    Matrix4 K;
    K.data[0] = make_float4(k.x, 0, k.z, 0);
    K.data[1] = make_float4(0, k.y, k.w, 0);
    K.data[2] = make_float4(0, 0, 1, 0);
    K.data[3] = make_float4(0, 0, 0, 1);
    return K;
}

inline Matrix4 getInverseCameraMatrix( const float4 & k ){
    Matrix4 invK;
    invK.data[0] = make_float4(1.0f/k.x, 0, -k.z/k.x, 0);
    invK.data[1] = make_float4(0, 1.0f/k.y, -k.w/k.y, 0);
    invK.data[2] = make_float4(0, 0, 1, 0);
    invK.data[3] = make_float4(0, 0, 0, 1);
    return invK;
}


template<typename P>
inline Matrix4 toMatrix4( const TooN::SE3<P> & p){
//    const TooN::Matrix<4, 4, float> I = TooN::Identity;
    Matrix4 R;
//    TooN::wrapMatrix<4,4>(&R.data[0].x) = p * I;
    return R;
}

inline Matrix4 inverse( const Matrix4 & A ){
//    static TooN::Matrix<4, 4, float> I = TooN::Identity;
//    TooN::Matrix<4,4,float> temp =  TooN::wrapMatrix<4,4>(&A.data[0].x);
    Matrix4 R;
//    TooN::wrapMatrix<4,4>(&R.data[0].x) = TooN::gaussian_elimination(temp , I );
    return R;
}

inline std::ostream & operator<<( std::ostream & out, const Matrix4 & m ){
    for(unsigned i = 0; i < 4; ++i)
        out << m.data[i].x << "  " << m.data[i].y << "  " << m.data[i].z << "  " << m.data[i].w << "\n";
    return out;
}

*/
