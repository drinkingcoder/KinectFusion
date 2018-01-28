#include "LocalConfigurator.h"

namespace {

__global__ void mm2meters(Image<float> out, const Image<ushort> in) {
    const uint2 pixel = thr2pos2();
    out[pixel] = in[pixel] / 6000.0f;
}

}

void LocalConfigurator::CUDATransfer(
        Image<float> out, Image<ushort> in
        ) {
//        CUDASafeCall(
        mm2meters<<<divup(m_parameters.InputSize, m_parameters.ImageBlock), m_parameters.ImageBlock>>>(
                out, in
                );
 //       );
}
