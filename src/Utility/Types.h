#pragma once

#include <Eigen/Core>
#include <Eigen/Eigen>
#include <Eigen/Geometry>

#include <cmath>

#include "cutil_math.h"

//#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

typedef Eigen::Affine3f Affine3f;

typedef Eigen::Matrix<float, 3, 4> Matrix3x4f;

typedef Eigen::Matrix<float, 6, 6> Matrix6f;
typedef Eigen::Matrix<float, 4, 4> Matrix4f;
typedef Eigen::Matrix<float, 3, 3> Matrix3f;

typedef Eigen::Matrix<float, 32, 1> Vector32f;
typedef Eigen::Matrix<float, 27, 1> Vector27f;
typedef Eigen::Matrix<float, 21, 1> Vector21f;
typedef Eigen::Matrix<float, 6, 1> Vector6f;
typedef Eigen::Matrix<float, 4, 1> Vector4f;
typedef Eigen::Matrix<float, 3, 1> Vector3f;

typedef Eigen::AngleAxisf AngleAxisf;
typedef Eigen::Quaternion<float> Quaternionf;

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

inline __host__ __device__ float3 operator*(const Matrix3x4f & M, const float3 &p) {
    float3 res;
    Eigen::Map<Vector3f> res_map(reinterpret_cast<float*>(&res));
    res_map = M.block(0, 0, 3, 3) * Eigen::Map<const Vector3f>(reinterpret_cast<const float*>(&p)) + M.block(0, 3, 3,  1);
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

inline __host__ __device__ Matrix4f combine_intrinsics(const Matrix3f & K, const Matrix4f & M) {
    Matrix4f res = Matrix4f::Identity();
    res.block(0, 0, 3, 3) = K * M.block(0, 0, 3, 3);
    res.block(0, 3, 3, 1) = K * M.block(0, 3, 3, 1);
    return res;
}

inline Matrix3f cross_dot_matrix(const Vector3f & v) {
    Matrix3f res = Matrix3f::Zero();
    res(0, 1) = -v(2);
    res(1, 0) = v(2);
    res(0, 2) = v(1);
    res(2, 0) = -v(1);
    res(1, 2) = -v(0);
    res(2, 1) = v(0);
    return res;
}

inline Matrix6f make_jtj(const Vector21f & v) {
    /*
    std::cout << "make_jtj Input Vector : " << std::endl;
    std::cout << v << std::endl;
    */
    Matrix6f res = Matrix6f::Zero();
    res.block(0, 0, 6, 1) = v.segment(0, 6);
    res.block(1, 1, 5, 1) = v.segment(6, 5);
    res.block(2, 2, 4, 1) = v.segment(11, 4);
    res.block(3, 3, 3, 1) = v.segment(15, 3);
    res.block(4, 4, 2, 1) = v.segment(18, 2);
    res(5, 5) = v(20);

    /*
    std::cout << "make_jtj before fill transpose : " << std::endl;
    std::cout << res << std::endl;
    */
    for (int r = 1; r < 6; r++) {
        for (int c = 0; c < r; c++) {
            res(c, r) = res(r, c);
        }
    }
    std::cout << "JTJ = " << std::endl;
    std::cout << res << std::endl;
    return res;
}

inline Vector6f solve(const Vector27f & v) {
    Vector6f b = v.segment(0, 6);
    Matrix6f M = make_jtj(v.segment(6, 21));

    return M.ldlt().solve(b);
}

// [translation-vector rotation-vector]
inline Matrix4f exp(const Vector6f & v) {
    Matrix4f res = Matrix4f::Zero();
    Vector3f rot_v = v.segment(3, 3);
    const float theta = rot_v.norm();
    rot_v = rot_v.normalized();

    Matrix3f J = sin(theta) / theta * Matrix3f::Identity()
                    + (1 - sin(theta) / theta) * rot_v * rot_v.transpose()
                    + (1 - cos(theta)) / theta * cross_dot_matrix(rot_v);

    std::cout << "rot_v = " << rot_v << std::endl;
    std::cout << "theta = " << theta << std::endl;
    std::cout << "sin theta = " << sin(theta) << std::endl;
    std::cout << "cos theta = " << cos(theta) << std::endl;
    std::cout << "Matrix J = " << std::endl;
    std::cout << J << std::endl;
    res.block(0, 3, 3, 1) = J * v.segment(0, 3);
    res.topLeftCorner(3, 3) = Matrix3f(AngleAxisf(theta, rot_v));
    res(3, 3) = 1;
    return res;
}

inline Matrix3f exp(const Vector3f & v) {
    return Matrix3f(AngleAxisf(v.norm(), v.normalized()));
}

inline Vector6f log(const Matrix4f & M) {
    Vector6f res;
    res.segment(0, 3) = M.block(0, 3, 3, 1);
    AngleAxisf av(Matrix3f(M.block(0, 0, 3, 3)));
    res.segment(3, 3) = av.axis()*av.angle();
    return res;
}

inline Vector3f log(const Matrix3f & M) {
    AngleAxisf v(M);
    return Vector3f(v.axis()*v.angle());
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
