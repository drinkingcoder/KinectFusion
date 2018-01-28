#include <memory>
//#include "TooN/se3.h"
#include "Visualizer.h"
#include "KinectFusion.h"

#define CONFIG_FILE "../config.json"

using namespace std;

void SetImageGreen(uchar4 *image, uint2 size) {
    for (auto i = 0; i < size.y; i++) {
        for (auto j = 0; j < size.x; j++) {
            auto offset = i * size.x + j;
            image[offset].x = 0;
            image[offset].y = 255;
            image[offset].z = 0;
        }
    }
}

LocalConfigurator *configurator;
Visualizer *visualizer;
Image<float3, HostDeviceAllocator> vertex, normal;
Image<float, HostDeviceAllocator> depth;
Image<uchar4, HostDeviceAllocator> rgb;
std::shared_ptr<KinectFusion> kinectfusion;
Matrix4f pose;

//SE3<float> preTrans(makeVector(0.0, 0, -0.9, 0, 0, 0));
//SE3<float> rot(makeVector(0.0, 0, 0, 0, 0, 0));
//SE3<float> trans(makeVector(0.5, 0.5, 0.5, 0, 0, 0));

void Run() {
    auto & params = configurator->m_parameters;
    Image<uchar4, HostDeviceAllocator> raw_rgb;
    Image<float, HostDeviceAllocator> raw_depth;
    raw_rgb.Allocate(params.InputSize);
    raw_depth.Allocate(params.InputSize);

    Matrix4f pose = Matrix4f::Identity();
    kinectfusion->SetPose(pose);

    Matrix4f pre_trans = Matrix4f::Identity();
    pre_trans.block(0, 3, 3, 1) = Vector3f(0, 0, -0.9);
    Matrix4f rot = Matrix4f::Identity();
    Matrix4f trans = Matrix4f::Identity();
    trans.block(0, 3, 3, 1) = Vector3f(0.5f, 0.5f, 0.5f);

    Matrix4f delta_rotation = Matrix4f::Identity();
    delta_rotation.topLeftCorner(3, 3) = Matrix3f(AngleAxisf(M_PI / 20, Vector3f::UnitX()));
    pose = trans * rot * pre_trans;

    kinectfusion->SetPose(pose);
    configurator->NextRGBAndDepthFrame(raw_rgb, raw_depth);
    kinectfusion->SetDepth(raw_depth);
    kinectfusion->Integrate();

    while(configurator->NextRGBAndDepthFrame(raw_rgb, raw_depth)) {
        kinectfusion->SetDepth(raw_depth);
        kinectfusion->Raycast();
        kinectfusion->Track();
        kinectfusion->Integrate();
//        rot *= delta_rotation;
        if (visualizer->Terminated()) {
            break;
        }
        cudaDeviceSynchronize();
        visualizer->UpdateWindowWithRGBAndDepthOnDevice(raw_rgb,
                kinectfusion->m_output,
                kinectfusion->m_vertex,
                kinectfusion->m_input_vertex[0]);
    }
}

int main(int argc, char ** argv) {
    // load configuration
    Parameters parameters;
    parameters.argc = argc;
    parameters.argv = argv;

    configurator = new LocalConfigurator();
    configurator->ParseParametersFromJsonFile(parameters, CONFIG_FILE);

    depth.Allocate(parameters.InputSize);
    rgb.Allocate(parameters.InputSize);
    vertex.Allocate(parameters.InputSize);
    normal.Allocate(parameters.InputSize);

    kinectfusion = make_shared<KinectFusion>(parameters);
//    kinectfusion->SetPose(toMatrix4(trans * rot * preTrans));

    Visualizer::SetParameters(parameters);
    visualizer->Setup();


//    SetImageGreen(rgb.m_data, parameters.InputSize);
//    visualizer->UpdateWindowWithRGBAndDepth(rgb.m_data, depth.m_data);

    auto t = std::thread(Run);

    visualizer->Run();

    cout <<"Finished." << endl;

    delete configurator;
    delete visualizer;
    return 0;
}

