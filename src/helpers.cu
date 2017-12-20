#include "helpers.h"
//#include "KinectFusion.h"

__global__ void RenderDepthKernel(Image<uchar3> out,
                                    const Image<float> depth,
                                    const float near_plane,
                                    const float far_plane) {
    const float d = (clamp(depth.el(), near_plane, far_plane) - near_plane) / (far_plane - near_plane);
    out.el() = make_uchar3(d * 255, d * 255, d * 255);
}

void RenderDepthMap(Image<uchar3> out, const Image<float> &depth, const float near_plane, const float far_plane) {
    dim3 block(32, 16);
    RenderDepthKernel<<<divup(depth.m_size, block), block>>>(out, depth, near_plane, far_plane);
}

__global__ void RenderNormalsKernel(Image<uchar3> out, const Image<float3> in) {
    float3 n = in.el();
    if(n.x == -2) {
        out.el() = make_uchar3(0, 0, 0);
    } else {
        n = normalize(n);
        out.el() = make_uchar3(n.x * 128 + 128, n.y * 128 + 128, n.z * 128 + 128);
    }
}

void RenderNormalMap(Image<uchar3> out, const Image<float3> &normal) {
    dim3 block(20, 20);
    RenderNormalsKernel<<<divup(normal.m_size, block), block>>>(out, normal);
}

__global__ void RenderTrackKernel(Image<uchar4> out, const Image<TrackData> data) {
    const uint2 pos = thr2pos2();
    switch(data[pos].result) {
        case 1:{
                   out[pos] = make_uchar4(128, 128, 128, 0);
                   break;
               }
        case -1:
        case -2:
        case -3:
        case -4:
        case -5:{
               out[pos] = make_uchar4(0, 0, 0, 0);
               break;
                }
    }
}

void RenderTrackResult(Image<uchar4> out, const Image<TrackData> &data) {
    dim3 block(32, 16);
    RenderTrackKernel<<<divup(out.m_size, block), block>>>(out, data);
}
