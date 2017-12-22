
#include <Eigen/Eigen>
#include <Eigen/Geometry>
#include <Eigen/Core>

#include "sophus/so3.hpp"
#include "sophus/se3.hpp"

#include <cmath>

typedef Eigen::Affine3f Affine3f;

typedef Eigen::Matrix<float, 4, 4> Matrix4f;
typedef Eigen::Matrix<float, 3, 3> Matrix3f;

typedef Eigen::Matrix<float, 6, 1> Vector6f;
typedef Eigen::Matrix<float, 4, 1> Vector4f;
typedef Eigen::Matrix<float, 3, 1> Vector3f;

typedef Eigen::AngleAxisf AngleAxisf;

int main() {
    float v[] = {1, 1, 1, 2, 2, 2, 3, 3, 3};
    auto m = Eigen::Map<Matrix3f>(v);
    auto m2 = Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>>(v);
    std::cout << m << std::endl;
    std::cout << m2 << std::endl;
}
