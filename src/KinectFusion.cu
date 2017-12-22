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

// TODO(): shared memory acceleration
__global__ void BilateralFilterKernel(
        Image<float> out,
        const Image<float> in,
        const Image<float> gaussian,
        const float sigma, //sigma for illuminance
        const int radius
        ) {
    const uint2 pix = thr2pos2();

    if (in[pix] == 0) {
        out[pix] = 0;
        return;
    }

    float sum = 0;
    float t = 0;
    const float center = in[pix];

    for (int i = -radius; i <= radius; i++) {
        for (int j = - radius; j <= radius; j++) {
//            const float pixel = in[make_uint2(clamp(pix.x + i, 0u, in.m_size.x - 1), clamp(pix.y + j, 0u, in.m_size.y - 1))];
            const float pixel = in[clamp(
                    make_uint2(pix.x + i, pix.y + j), 
                    make_uint2(0u, 0u),
                    make_uint2(in.m_size.x - 1, in.m_size.y - 1)
                    )];
            if (pixel > 0) {
                const float mod = sqr(pixel - center);
                const float factor = gaussian[make_uint2(i + radius, 0)] * gaussian[make_uint2(j + radius, 0)] * __expf(-mod / (2 * sqr(sigma)));
                t += factor * pixel;
                sum += factor;
            }
        }
    }
    out[pix] = t / sum;
}

__global__ void HalfSampleRobustKernel(
        Image<float> out,
        const Image<float> in,
        const float robus_threshold,
        const int radius
        ) {
    const uint2 pix = thr2pos2();
    const uint2 center_pix = 2 * pix;

    if (pix.x >= out.m_size.x || pix.y >= out.m_size.y) {
        return;
    }

    float sum = 0;
    float t = 0;
    const float center_pixel = in[center_pix];

    for (int i = -radius + 1; i <= radius; i++) {
        for (int j = - radius + 1; j <= radius; j++) {
            const uint2 pix_in = clamp(
                        make_uint2(center_pix.x + i, center_pix.y + j),
                        make_uint2(0u, 0u),
                        make_uint2(in.m_size.x - 1, in.m_size.y - 1)
                        );
            const float pixel = in[pix_in];
            if (fabsf(pixel - center_pixel) < robus_threshold) { //accept  illuminance within neighberhood in threshold
                sum += 1;
                t += pixel;
            }
        }
    }
    out[pix] = t / sum;
}

// gaussian for space difference
__global__ void GenerateGaussianKernel(Image<float> out, float sigma, int radius) {
    int x = threadIdx.x - radius;
    out[make_uint2(threadIdx.x, 0)] = __expf(-sqr(x) / (2 * sqr(sigma)));
}

__global__ void Depth2VertexKernel(Image<float3> vertex, const Image<float> depth, const Matrix3f invK) {
    const uint2 pix = thr2pos2();
    if (pix.x >= depth.m_size.x || pix.y >= depth.m_size.y) {
        return;
    }

    if (depth[pix] > 0) {
        vertex[pix] = depth[pix] * (invK * make_float3(pix.x, pix.y, 1));
    } else {
        vertex[pix] = make_float3(0);
    }
}

__global__ void Vertex2NormalKernel(Image<float3> normal, const Image<float3> vertex) {
    const uint2 pix = thr2pos2();
    if (pix.x >= vertex.m_size.x || pix.y >= vertex.m_size.y) {
        return;
    }

    const float3 left = vertex[make_uint2(max(static_cast<int>(pix.x) - 1, 0), pix.y)];
    const float3 right = vertex[make_uint2(min(pix.x + 1, vertex.m_size.x - 1), pix.y)];
    const float3 up = vertex[make_uint2(pix.x, max(static_cast<int>(pix.y) - 1, 0))];
    const float3 down = vertex[make_uint2(pix.x, min(pix.y + 1, vertex.m_size.y - 1))];

    if (left.z == 0 || right.z == 0 || up.z == 0 || down.z == 0) {
        normal[pix].x = INVALID;
        return;
    }

    const float3 dxv = right - left;
    const float3 dyv = down - up;
    normal[pix] = normalize(cross(dyv, dxv));
}

/*!
 * f = sqr(n^T(p - p'))
 * e = n^T(p - p')
 * r = p - p'
 * J_r = ( I -p'^ )
 * J_e = n^TJ_r = n^T( I -p'^ )
 */
__global__ void TrackKernel(
        Image<TrackData> out,
        const Image<float3> input_vertex,
        const Image<float3> input_normal,
        const Image<float3> ref_vertex,
        const Image<float3> ref_normal,
        const Matrix4f track_pose,
        const Matrix4f K_inv_raycast_pose,
        const float dist_threshold,
        const float normal_threshold
        ) {
    const uint2 pix = thr2pos2();
    if (pix.x >= input_vertex.m_size.x || pix.y >= input_normal.m_size.y) {
        return;
    }

    TrackData & row = out[pix];

    if (input_normal[pix].x == INVALID) {
        row.result = -1;
        return;
    }

    // find corresponding vertex from input_vertex to ref_vertex
    // project vertex from current camera to world
    const float3 projected_vertex = transform(track_pose, input_vertex[pix]);
    // project vertex from world to image with raycast pose
    const float3 projected_pos = transform(K_inv_raycast_pose, projected_vertex);
    // get image position
    const float2 projected_pix = make_float2(
            projected_pos.x / projected_pos.z + 0.5f,
            projected_pos.y / projected_pos.z + 0.5f
            );

    // out of border
    if (projected_pix.x < 0 || 
            projected_pix.x >= ref_vertex.m_size.x ||
            projected_pix.y < 0 ||
            projected_pix.y >= ref_vertex.m_size.y) {
        row.result = -2;
        return;
    }

    const uint2 ref_pix = make_uint2(projected_pix.x, projected_pix.y);
    const float3 ref_normal_pixel = ref_normal[ref_pix];

    if (ref_normal_pixel.x == INVALID) {
        row.result = -3;
        return;
    }

    // Euclidean difference
    const float3 diff = ref_vertex[ref_pix] - projected_vertex;
    // project normal from current camera to world
    const float3 projected_normal = rotate(track_pose, input_normal[pix]);

    // outlier
    if (length(diff) > dist_threshold) {
        row.result = -4;
        return;
    }

    //TODO: it seems that projected_normal is in world, but ref_normal is in raycast camera
    //if the normal is figured out from TSDF, it's OK.
    if (dot(projected_normal, ref_normal_pixel) < normal_threshold) {
        row.result = -5;
        return;
    }

    row.result = 1;
    row.error = dot(ref_normal_pixel, diff);
    reinterpret_cast<float3 *>(row.J)[0] = ref_normal_pixel; // N
    reinterpret_cast<float3 *>(row.J)[1] = cross(projected_vertex, ref_normal_pixel); // N dot (-p'^)
}

//TODO: it can be accelerated by reduce primitive
__global__ void ReduceKernel(float * out, const Image<TrackData> J, const uint2 size){
    __shared__ float S[112][32]; // this is for the final accumulation
    const uint sline = threadIdx.x;

    float sums[32]; // 1 for error, 21 for jtj, 6 for jte, 4 for vertex info
    float * jtj = sums + 7;
    float * info = sums + 28;

    for(uint i = 0; i < 32; ++i)
        sums[i] = 0;

    for(uint y = blockIdx.x; y < size.y; y += gridDim.x){
        for(uint x = sline; x < size.x; x += blockDim.x ){
            const TrackData & row = J[make_uint2(x, y)];
            if(row.result < 1){
                info[1] += row.result == -4 ? 1 : 0;
                info[2] += row.result == -5 ? 1 : 0;
                info[3] += row.result > -4 ? 1 : 0;
                continue;
            }

            // Error part
            sums[0] += row.error * row.error;

            // JTe part
            for(int i = 0; i < 6; ++i)
                sums[i+1] += row.error * row.J[i];

            // JTJ part, unfortunatly the double loop is not unrolled well...
            jtj[0] += row.J[0] * row.J[0];
            jtj[1] += row.J[0] * row.J[1];
            jtj[2] += row.J[0] * row.J[2];
            jtj[3] += row.J[0] * row.J[3];
            jtj[4] += row.J[0] * row.J[4];
            jtj[5] += row.J[0] * row.J[5];

            jtj[6] += row.J[1] * row.J[1];
            jtj[7] += row.J[1] * row.J[2];
            jtj[8] += row.J[1] * row.J[3];
            jtj[9] += row.J[1] * row.J[4];
           jtj[10] += row.J[1] * row.J[5];

           jtj[11] += row.J[2] * row.J[2];
           jtj[12] += row.J[2] * row.J[3];
           jtj[13] += row.J[2] * row.J[4];
           jtj[14] += row.J[2] * row.J[5];

           jtj[15] += row.J[3] * row.J[3];
           jtj[16] += row.J[3] * row.J[4];
           jtj[17] += row.J[3] * row.J[5];

           jtj[18] += row.J[4] * row.J[4];
           jtj[19] += row.J[4] * row.J[5];

           jtj[20] += row.J[5] * row.J[5];

           // extra info here
           info[0] += 1;
        }
    }

    for(int i = 0; i < 32; ++i) // copy over to shared memory
        S[sline][i] = sums[i];

    __syncthreads();            // wait for everyone to finish

    if(sline < 32){ // sum up columns and copy to global memory in the final 32 threads
        for(unsigned i = 1; i < blockDim.x; ++i)
            S[0][sline] += S[i][sline];
        out[sline+blockIdx.x*32] = S[0][sline];
    }
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

    m_output.Allocate(parameters.InputSize);
    m_ba_values.Allocate(make_uint2(32, 8));

    m_raw_depth.Allocate(parameters.InputSize);
    m_vertex.Allocate(parameters.InputSize);
    m_normal.Allocate(parameters.InputSize);

    m_cameraK = m_parameters.CameraK;

    m_input_depth.resize(parameters.ICPLevels);
    m_input_vertex.resize(parameters.ICPLevels);
    m_input_normal.resize(parameters.ICPLevels);
    m_inv_cameraKs.resize(parameters.ICPLevels);
    for (auto i = 0; i < parameters.ICPLevels; i++) {
        m_input_depth[i].Allocate(parameters.InputSize >> i);
        m_input_vertex[i].Allocate(parameters.InputSize >> i);
        m_input_normal[i].Allocate(parameters.InputSize >> i);

        m_inv_cameraKs[i] << (1 << i) / m_cameraK(0, 0) , 0, (1 << i) / m_cameraK(0, 2),
                                0, (1 << i) / m_cameraK(1, 1), (1 << i) / m_cameraK(1, 2),
                                0, 0, 1;
    }

    m_gaussian.Allocate(make_uint2(parameters.GaussianRadius * 2 + 1, 1));

    GenerateGaussianKernel<<<1, m_gaussian.m_size.x>>>(
            m_gaussian,
            parameters.GaussianFunctionSigma,
            parameters.GaussianRadius
            );
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
    m = m_raycast_pose * m.inverse();
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

/*!
 * 1.bilateral filter and down sample(coarse to fine)
 * 2.Coarse to fine iteration:
 *   1.compute vertex and normal from raycast
 *   2.compute jaccobian
 *   3.reduce jaccobian and solve linear equation
 *   4.update pose, goto 1(optimize iteration)
 */
bool KinectFusion::Track() {
    std::vector<dim3> grids;
    for (auto i = 0; i < m_parameters.ICPLevels; i++) {
        grids.push_back(divup(m_parameters.InputSize >> i, m_parameters.ImageBlock));
    }

    // bilateral filter
    BilateralFilterKernel<<<grids[0], m_parameters.ImageBlock>>>(
            m_input_depth[0],
            m_raw_depth,
            m_gaussian,
            m_parameters.GaussianIlluminanceSigma,
            m_parameters.GaussianRadius
            );

    // downsample
    for (auto i = 1; i < m_parameters.ICPLevels; i++) {
        HalfSampleRobustKernel<<<grids[i], m_parameters.ImageBlock>>>(
                m_input_depth[i],
                m_input_depth[i-1],
                m_parameters.GaussianIlluminanceSigma * 3,
                m_parameters.GaussianRadius
                );
    }
    for (auto itr = 0; itr < m_parameters.ICPLevels; itr++) {
        Depth2VertexKernel<<<grids[itr], m_parameters.ImageBlock>>>(
                m_input_vertex[itr],
                m_input_depth[itr],
                m_inv_cameraKs[itr]
                );
        Vertex2NormalKernel<<<grids[itr], m_parameters.ImageBlock>>>(
                m_input_normal[itr],
                m_input_vertex[itr]
                );
    }

    const Matrix4f old_pose = m_pose;
    const Matrix4f inv_raycast_pose = m_raycast_pose.inverse();
    const Matrix4f project_ref = combine_intrinsics(m_cameraK, inv_raycast_pose);

    auto values = Eigen::Map<Eigen::Matrix<float, 8, 32, Eigen::RowMajor>>(static_cast<float*>(m_ba_values.m_data));
    for (auto level = m_parameters.ICPLevels - 1; level >= 0; level--) {
        for (auto itr = 0; itr < m_parameters.ICPIterationTimes[level]; itr++) {
            TrackKernel<<<grids[level], m_parameters.ImageBlock>>>(
                    m_reduction,
                    m_input_vertex[level],
                    m_input_normal[level],
                    m_vertex,
                    m_normal,
                    m_pose,
                    project_ref,
                    0.2f, // dist_threshold
                    0.8f  // normal_threshold
                    );
            ReduceKernel<<<8, 112>>>(
                    m_ba_values.GetDeviceImage().Data(),
                    m_reduction,
                    m_input_vertex[level].m_size
                    );

            cudaDeviceSynchronize(); //synchronize host pin memory
            // solve linear equation
            Vector32f v = values.colwise().sum();
            Vector6f delta_se3x = solve(Vector27f(v.segment(1, 27)));

            // SE3 transform
            Matrix3f R = AngleAxisf(delta_se3x.segment(3, 3).norm(), delta_se3x.segment(3, 3));
            m_pose = Sophus::SE3<float>::exp(R, Vector3f(delta_se3x.segment(0, 3))).matrix() * m_pose;

            if (delta_se3x.norm() < 1e-5) {
                std::cout << "Ahead of optimization" << std::endl;
                break;
            }
        }
    }

    if ((sqrt(values(0, 0) / values(0, 28)) > 2e-2) || (values(0, 28)) / (m_raw_depth.m_size.x * m_raw_depth.m_size.y) < 0.15f) {
        std::cout << "Don't update pose" << std::endl;
        m_pose = old_pose;
        return false;
    }
    return true;
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
