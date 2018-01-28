#pragma once

#include "Configurator/Configurator.h"
#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
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
//    char FrameIndexFilledWithCharacter;
//    int FrameIndexWidth;

    int GaussianRadius = 2;
    float GaussianFunctionSigma = 4.0f;
    float GaussianIlluminanceSigma = 0.1f;

//    std::string CalibrationFile,
    std::string FramePath;
    std::string RGBConfigFile, DepthConfigFile;
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
        parameters.CameraK(0, 2) = 320.1;
        parameters.CameraK(1, 2) = 247.6;
        parameters.CameraK(0, 0) = 535.4;
        parameters.CameraK(1, 1) = 539.2;
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

        /*
        auto tmpstring = m_json_parser->GetString("FrameIndexFilledWithCharacter");
        parameters.FrameIndexFilledWithCharacter = tmpstring[0];
        parameters.FrameIndexWidth = m_json_parser->GetInt("FrameIndexWidth");
        */

//        parameters.CalibrationFile = m_json_parser->GetString("CalibrationFile");
        parameters.FramePath = m_json_parser->GetString("FramePath");
        parameters.RGBConfigFile = m_json_parser->GetString("RGBConfigFile");
        parameters.DepthConfigFile = m_json_parser->GetString("DepthConfigFile");

        m_frame_count = parameters.FrameIndexHead;
        tmp_depth.Allocate(parameters.InputSize);

        parameters.ImageBlock = dim3(32, 16);

        parameters.ICPIterationTimes.resize(3);
        parameters.ICPIterationTimes[0] = 20;
        parameters.ICPIterationTimes[1] = 10;
        parameters.ICPIterationTimes[2] = 10;

        std::string fn = parameters.FramePath+parameters.RGBConfigFile;
        m_rgbf.open(fn);
        if( !m_rgbf ) {
            std::cout << "Can not open RGB Config File." <<std::endl;
            exit(-1);
        }
        std::cout << "RGB Config content" << std::endl;
        for (auto i = 0; i < 3; i++) {
            std::string s;
            std::getline(m_rgbf, s);
            std::cout << s;
        }
        fn = parameters.FramePath+parameters.DepthConfigFile;
        m_depthf.open(fn);
        if( !m_depthf ) {
            std::cout << "Can not open Depth Config File." <<std::endl;
            exit(-1);
        }
        std::cout << "Depth Config content" << std::endl;
        for (auto i = 0; i < 3; i++) {
            std::string s;
            std::getline(m_depthf, s);
            std::cout << s;
        }

        m_parameters = parameters;
	}

    // rgb - uchar4
    // depth - float
    bool NextRGBAndDepthFrame(Image<uchar4> rgb, Image<float> depth) {
        if (m_frame_count > m_parameters.FrameIndexTail) {
            std::cout << "End of index count." << std::endl;
            return false;
        }

        float timestamp;
        std::string fn;
        m_depthf >> timestamp >> fn;
        fn = m_parameters.FramePath + fn;

        cv::Mat read_image = cv::imread(fn, -1);
        if (! read_image.data ) {
            std::cout << "Failed: File <" << fn << ">" << std::endl;
            std::cout << "    Not Found! " << std::endl;
            exit(0);
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

        m_rgbf >> timestamp >> fn;
        fn = m_parameters.FramePath + fn;

        read_image = cv::imread(fn, -1);
        if (! read_image.data ) {
            std::cout << "Failed: File <" << fn << ">" << std::endl;
            std::cout << "    Not Found! " << std::endl;
            exit(0);
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
    std::ifstream m_rgbf, m_depthf;
};
