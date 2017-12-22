#pragma once

#include "Configurator/Configurator.h"
#include <cuda_runtime.h>
#include <iostream>
#include <utility>
#include "Types.h"
#include "cutil_math.h"
#include "CudaToolKit.h"
#include <sstream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

struct Parameters {
    int argc;
    char **argv;
	std::string ConfigurationName;

    Matrix3f CameraK;
    float NearPlane, FusionThreshold;
    float FarPlane = 4.0f;
    float MaxWeight;
    uint3 VolumeSize;
    float3 VolumeDimensions;
    uint2 InputSize;
	std::string RGBFramePrefix, RGBFrameSuffix, DepthFramePrefix, DepthFrameSuffix;
	int FrameIndexHead, FrameIndexTail;
    char FrameIndexFilledWithCharacter;
    int FrameIndexWidth;

    int GaussianRadius = 2;
    float GaussianFunctionSigma = 4.0f;
    float GaussianIlluminanceSigma = 0.1f;

    std::string CalibrationFile, FramePath;
    dim3 ImageBlock;

    std::vector<int> ICPIterationTimes;
    int ICPLevels = 3;

};

class LocalConfigurator: public Configurator<Parameters> {
public:
    LocalConfigurator():Configurator() {
    }
    ~LocalConfigurator () {
    }

	void ParseParametersFromJsonFile(Parameters &parameters, const char *fname) override {
		m_valid = m_json_parser->ParseFile(fname);
		if (!m_valid) {
			return;
		}

		parameters.ConfigurationName = std::string(fname);

        parameters.MaxWeight = m_json_parser->GetFloat("MaxWeight");

        /*
        auto vf = m_json_parser->GetFloatVec("CameraK");
        */
        parameters.CameraK = Matrix3f::Identity();
        parameters.CameraK(0, 2) = 346.471;
        parameters.CameraK(1, 2) = 249.031;
        parameters.CameraK(0, 0) = 573.71;
        parameters.CameraK(1, 1) = 574.394;
        /*
        parameters.CameraK(0, 2) = 160;
        parameters.CameraK(1, 2) = 120;
        parameters.CameraK(0, 0) = 100;
        parameters.CameraK(1, 1) = 100;
        */

        parameters.NearPlane = m_json_parser->GetFloat("NearPlane");
        parameters.FusionThreshold = m_json_parser->GetFloat("FusionThreshold");

        auto vi = m_json_parser->GetIntVec("VolumeSize");
        parameters.VolumeSize = make_uint3(vi[0], vi[1], vi[2]);
        parameters.VolumeDimensions = make_float3(1.0f, 1.0f, 1.0f);

        vi = m_json_parser->GetIntVec("InputSize");
        parameters.InputSize = make_uint2(vi[0], vi[1]);

		parameters.RGBFramePrefix = m_json_parser->GetString("RGBFramePrefix");
		parameters.RGBFrameSuffix = m_json_parser->GetString("RGBFrameSuffix");
		parameters.DepthFramePrefix = m_json_parser->GetString("DepthFramePrefix");
		parameters.DepthFrameSuffix = m_json_parser->GetString("DepthFrameSuffix");

        parameters.FrameIndexHead = m_json_parser->GetInt("FrameIndexHead");
        parameters.FrameIndexTail = m_json_parser->GetInt("FrameIndexTail");

        auto tmpstring = m_json_parser->GetString("FrameIndexFilledWithCharacter");
        parameters.FrameIndexFilledWithCharacter = tmpstring[0];
        parameters.FrameIndexWidth = m_json_parser->GetInt("FrameIndexWidth");

        parameters.CalibrationFile = m_json_parser->GetString("CalibrationFile");
        parameters.FramePath = m_json_parser->GetString("FramePath");

        m_frame_count = parameters.FrameIndexHead;
        m_parameters = parameters;
        tmp_depth.Allocate(parameters.InputSize);

        m_parameters.ImageBlock = dim3(32, 16);

        m_parameters.ICPIterationTimes.resize(3);
        m_parameters.ICPIterationTimes[0] = 10;
        m_parameters.ICPIterationTimes[1] = 5;
        m_parameters.ICPIterationTimes[2] = 5;
	}

    // rgb - uchar4
    // depth - float
    bool NextRGBAndDepthFrame(Image<uchar4> rgb, Image<float> depth) {
        if (m_frame_count > m_parameters.FrameIndexTail) {
            std::cout << "End of index count." << std::endl;
            return false;
        }

        m_frame_ss.width(m_parameters.FrameIndexWidth);
        m_frame_ss.fill(m_parameters.FrameIndexFilledWithCharacter);
        m_frame_ss.str("");
        m_frame_ss << m_frame_count;
        m_frame_count++;

        auto fn = m_parameters.FramePath +
                    m_parameters.DepthFramePrefix +
                    m_frame_ss.str() +
                    m_parameters.DepthFrameSuffix;

        cv::Mat read_image = cv::imread(fn, -1);
        if (! read_image.data ) {
            std::cout << "Failed: File <" << fn << ">" << std::endl;
            std::cout << "    Not Found! " << std::endl;
            return false;
        }
        CUDASafeCall(
                cudaMemcpy(
                    tmp_depth.m_data,
                    read_image.data,
                    m_parameters.InputSize.x * m_parameters.InputSize.y * sizeof(ushort),
                    cudaMemcpyHostToDevice
                    )
                );
        CUDATransfer(depth, tmp_depth);

        fn = m_parameters.FramePath +
                    m_parameters.RGBFramePrefix +
                    m_frame_ss.str() +
                    m_parameters.RGBFrameSuffix;

        read_image = cv::imread(fn, -1);
        if (! read_image.data ) {
            std::cout << "Failed: File <" << fn << ">" << std::endl;
            std::cout << "    Not Found! " << std::endl;
            return false;
        }
        cv::Mat rgb_image(read_image.size(), CV_MAKE_TYPE(read_image.type(), 4));
        // Notice: convertTo can't change the channels of Mat
        int from_to[] = {0, 0, 1, 1, 2, 2, -1, 3};
        cv::mixChannels(&read_image, 1, &rgb_image, 1, from_to, 4);

        CUDASafeCall(
                cudaMemcpy(
                    rgb.m_data,
                    rgb_image.data,
                    m_parameters.InputSize.x * m_parameters.InputSize.y * 4,
                    cudaMemcpyHostToDevice
                    )
                );
        return true;
    }

    void CUDATransfer(
            Image<float> out, Image<ushort> in
            );

    Parameters m_parameters;
private:
    int m_frame_count;
    std::stringstream m_frame_ss;
    Image<ushort, DeviceAllocator> tmp_depth;
};
