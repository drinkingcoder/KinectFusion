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

class KinectFusion {
public:
    Parameters m_parameters;
    Volume m_volume;
    Image<TrackData, DeviceAllocator> m_reduction;
    Image<float3, DeviceAllocator> m_vertex, m_normal;

    Image<float, DeviceAllocator> m_raw_depth;
    Image<float, HostDeviceAllocator> m_output;

    Matrix4f m_pose, m_raycast_pose;
    Matrix3f m_cameraK;

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

    /// just raycast
    void Raycast();
    /// figure out the raycast map originated from view for visualization
    void RenderInput(Image<float3> pos3D,
//                        Image<float3> normal,
                        Image<float> depth,
                        const Volume volume,
                        const Matrix4f view,
                        const float near_plane,
                        const float far_plane,
                        const float step,
                        const float large_step);

    float StepSize() const {
        return 0.007;
        return min(m_parameters.VolumeDimensions) / max(m_parameters.VolumeSize);
    }
    float LargeStepSize() const {
//        return 0.007;
        return 0.01;
        return 0.7f * m_parameters.FusionThreshold;
    }
protected:
private:
};
