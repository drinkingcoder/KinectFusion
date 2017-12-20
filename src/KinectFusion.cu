#include "KinectFusion.h"

#undef isnan
#undef isfinite

#include <iostream>

using namespace std;

namespace {
///Paper: Interactive Ray Tracing for Isosurface Rendering
__device__ __forceinline__ float4 RaycastForPos(const Volume volume,
                                            const uint2 pos,
                                            const Matrix4f view, ///< camera to world
                                            const float near_plane,
                                            const float far_plane,
                                            const float step,
                                            const float large_step) {
    const float3 origin = make_float3(view(0, 3), view(1, 3), view(2, 3));
    const float3 direction  = rotate(view, make_float3(pos.x, pos.y, 1));

    // intersect ray with a box, compute intersection of ray with six box planes
    // x = dx*t + px
    // y = dy*t + py
    // z = dz*t + pz
    // t means the length of the ray from origin to (x, y, z)
    const float3 invR = make_float3(1.0f) / direction; ///< 1 / direction
    /// make x = 0, then we'll get equation 0 = dx*t + px -> t = - px/dx
    const float3 bot = -1 * invR * origin;
    /// make x = dimx, then we'll get equation dimx = dx*t + px -> t =  (dimx - px) / dx
    const float3 top = invR * (volume.m_dim - origin);
    ///so bot stores length of the ray with (0, 0, 0) endpoint
    //top stores length of the ray with (dimx, dimy, dimz) endpoint

    // reorder intersections to find smallest and largest on each axis
    const float3 tmin = fminf(bot, top);
    const float3 tmax = fmaxf(bot, top);

    const float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    const float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    // get the nearest and the most far distance for cull
    //const float tnear = fmaxf(largest_tmin. near_plane); // why error here? the dot '.'
    const float tnear = fmaxf(largest_tmin, near_plane);
    const float tfar = fminf(smallest_tmax, far_plane);
    if (tnear < tfar) {
        // first walk with large_step to find a hit
        float t = tnear;
        float step_size = large_step;
        float f_t = volume.interp(origin + direction * t);
        float f_tt = 0;
        if (f_t > 0) {
            for (; t < tfar; t += step_size ) {
                f_tt = volume.interp(origin + direction * t);
                if (f_tt < 0) { // hit found
                    break;
                }
                if (f_tt < 0.8f) { //change to a smaller step_size
                    step_size = step;
                }
                f_t = f_tt;
            }
            if (f_tt < 0) { // hit found
                t = t + step_size * f_tt / (f_t - f_tt); // interpolation
                return make_float4(origin + direction * t, t); /// x = px + dx * t
            }
        }
    }
    return make_float4(0);
}

inline __device__ float sqr(const float x) {
    return x * x;
}

__global__ void IntegrateKernel(Volume volume,
                            const Image<float> depth,
                            const Matrix4f inv_track,
                            const Matrix3f K,
                            const float mu,
                            const float max_weight
        ) {
    uint3 pix = make_uint3(thr2pos2());
    //'auto' inference will result in deadly memory illegal access!
    float3 pos = transform(inv_track, volume.pos(pix));
    float3 cameraX = K * pos; // (u, v, 0) in image coordinate

    // each increment `delta` is a voxel on z-axis in fact.
    const float3 delta = rotate(inv_track, make_float3(0, 0, volume.m_dim.z / volume.m_size.z));
    const float3 camera_delta = K * delta;

    // each thread manipulate a column
    for (pix.z = 0; pix.z < volume.m_size.z; pix.z++, pos += delta, cameraX += camera_delta) {
        if (pos.z < 0.001f) {
            continue;
        }
        //project the voxel on image plane
        //why external 0.5?
        const float2 pixel = make_float2(cameraX.x / cameraX.z + 0.5f, cameraX.y / cameraX.z + 0.5f);
        if (pixel.x < 0 || pixel.x >= depth.m_size.x || pixel.y < 0 || pixel.y >= depth.m_size.y) {
            continue;
        }
        const uint2 px = make_uint2(pixel.x , pixel.y);
        if (depth[px] == 0) {
            continue;
        }
        const float diff = (depth[px] - cameraX.z) * (1 +sqr(pos.x / pos.z) + sqr(pos.y / pos.z));
        //depth on image is vertical depth, so we use Z rather that length of ray
        // scalar = (sqr(x) + sqr(y) + sqr(z))/sqr(z)
        // diff = distance * sqr(atan(theta)) - a approximation of projective diff

        //update the voxel when depth difference less than threshold
        if (diff > -mu) {
            const float sdf = fminf(1.0f, diff/mu);
            float2 data = volume[pix]; //x = tsdf, y = weight
            data.x = clamp((data.y * data.x + sdf) / (data.y + 1), -1.0f, 1.0f);
            data.y = fminf(data.y + 1, max_weight);
            volume.set(pix, data);
        }
    }
}

/// this kernel just figure out the hit-map: pos3D
/// output: pos3D gives the voxel in world space
/*
__global__ void RaycastKernel( Image<float3> pos3D,
                    Image<float3> normal,
                    const Volume volume,
                    const Matrix4 view,
                    const float near_plane,
                    const float far_plane,
                    const float step,
                    const float large_step) {
    const uint2 pos = thr2pos2();
    const float4 hit = RaycastForPos(volume, pos, view, near_plane, far_plane, step, large_step);
    if (hit.w > 0) {
        pos3D[pos] = make_float3(hit);
    } else {
        pos3D[pos] = make_float3(0);
    }
}
*/
}

/// this kernel gives an externel depth map for visualization
__global__ void RaycastDepthImageKernel(
                Image<float3> pos3D,
//                Image<float3> normal,
                Image<float> depth,
                Volume volume,
                Matrix4f view,
                float near_plane,
                float far_plane,
                float step,
                float large_step) {
    const auto pos = thr2pos2();

    float4 hit = RaycastForPos(
            volume,
            pos,
            view,
            near_plane,
            far_plane,
            step,
            large_step
            );
    if (hit.w > 0) {
//        printf("Hit!(%d, %d) = %f\n", pos.x, pos.y, hit.w);
        pos3D[pos] = make_float3(hit);
        depth[pos] = hit.w;
    } else {
        pos3D[pos] = make_float3(0);
        depth[pos] = 0;
    }
}

KinectFusion::KinectFusion(const Parameters& parameters):
                                m_parameters(parameters) {

    m_volume.Init(parameters.VolumeSize, parameters.VolumeDimensions);
//    m_volume.SetBoxWrap(make_float3(0.1f, 0.1f, 0.8f), make_float3(0.9f, 0.9f, 0.9f), -1.0f);
    m_volume.SetBoxWrap(make_float3(0.1f, 0.8f, 0.1f), make_float3(0.9f, 0.9f, 0.9f), -1.0f);
//    m_volume.SetBoxWrap(make_float3(0.8f, 0.1f, 0.1f), make_float3(0.9f, 0.9f, 0.9f), -1.0f);

    cudaSetDeviceFlags(cudaDeviceMapHost);
    m_raw_depth.Allocate(parameters.InputSize);
    m_output.Allocate(parameters.InputSize);
    m_vertex.Allocate(parameters.InputSize);
    m_cameraK = m_parameters.CameraK;
}

void KinectFusion::Raycast() {
    m_raycast_pose = m_pose;
    dim3 block(16, 16);
    std::cout << "near plane = " << m_parameters.NearPlane << std::endl;
    std::cout << "far plane = " << m_parameters.FarPlane << std::endl;
    std::cout << "step size = " << StepSize() << std::endl;
    std::cout << "large step size = " << LargeStepSize() << std::endl;
    std::cout << "Pose = " << std::endl;
    Matrix4f m = Matrix4f::Identity();
    m(0, 0) = m_cameraK(0, 0);
    m(1, 1) = m_cameraK(1, 1);
    m(0, 2) = m_cameraK(0, 2);
    m(1, 2) = m_cameraK(1, 2);
    m = m_pose * m.inverse();
    std::cout << m << std::endl;

    RaycastDepthImageKernel<<<divup(m_parameters.InputSize, block), block>>>(
            m_vertex,
            m_output.GetDeviceImage(),
            m_volume,
            m,
            m_parameters.NearPlane,
            m_parameters.FarPlane,
            StepSize(),
            LargeStepSize()
            );
}

void KinectFusion::Integrate() {
    Matrix4f inverse_pose = m_pose.inverse();
    std::cout << "inverse pose:" << std::endl << inverse_pose << std::endl;
    IntegrateKernel<<<divup(dim3(m_volume.m_size.x, m_volume.m_size.y), m_parameters.ImageBlock), m_parameters.ImageBlock>>>(
            m_volume,
            m_raw_depth,
            inverse_pose,
            m_parameters.CameraK,
            m_parameters.FusionThreshold,
            m_parameters.MaxWeight
            );
}

void KinectFusion::RenderInput(Image<float3> pos3D,
//        Image<float3> normal,
        Image<float> depth,
        const Volume volume,
        const Matrix4f view,
        const float near_plane,
        const float far_plane,
        const float step,
        const float large_step) {
    dim3 block(16, 16);
    RaycastDepthImageKernel<<<divup(pos3D.m_size, block), block>>>(
            pos3D,
//            normal,
            m_output,
            volume,
            view,
            near_plane,
            far_plane,
            step,
            large_step
            );
}
