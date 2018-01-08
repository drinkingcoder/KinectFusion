#include <iostream>
#include <vector>
#include <stdint.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <cuda_runtime.h>

#include "Utility/cutil_math.h"
#include "Utility/CudaToolKit.h"
#include "Utility/LocalConfigurator.h"
#include "Volume.h"

#define INVALID -2

class KinectFusion {
public:
    Parameters m_parameters;
    Volume m_volume;
    Image<TrackData, DeviceAllocator> m_reduction;
    Image<float, DeviceAllocator> m_raw_depth;
    Image<float3, DeviceAllocator> m_vertex, m_normal;

    std::vector<Image<float, DeviceAllocator>> m_input_depth;
    std::vector<Image<float3, DeviceAllocator>> m_input_normal, m_input_vertex;

    Image<float, HostDeviceAllocator> m_output;
    Image<float, HostDeviceAllocator> m_ba_values;

    Image<float, DeviceAllocator> m_gaussian;

    Matrix4f m_pose, m_raycast_pose;
    Matrix3f m_cameraK;
    std::vector<Matrix3f> m_inv_cameraKs;

    KinectFusion(const Parameters& parameters);
    virtual ~KinectFusion() {

    }
    void SetPose(const Matrix4f & p) {
        m_pose = p;
    }

    template<typename T>
        void SetDepth(const Image<float, T> &depth) {
            m_raw_depth = depth;
        }

    void Integrate();
    bool Track();

    /// just raycast
    void Raycast();

    float StepSize() const {
        return min(m_parameters.VolumeDimensions) / max(m_parameters.VolumeSize);
    }
    float LargeStepSize() const {
//        return 0.007;
        return 0.01;
//        return 0.7f * m_parameters.FusionThreshold;
    }
protected:
private:
};
