#pragma once

#include<cuda_gl_interop.h>
#include "Utility/cutil_math.h"
#include "Utility/CudaToolKit.h"

/// scale near plane and far plane to 0-1
void RenderDepthMap(Image<uchar3> out, const Image<float> &, const float near_plane, const float far_plane);
/// render normal map to RGB
void RenderNormalMap(Image<uchar3> out, const Image<float3> &);
void RenderTrackResult(Image<uchar4> out, const Image<TrackData> &data);
