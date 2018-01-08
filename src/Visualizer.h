#pragma once

#include <limits>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <thread>
#include <mutex>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#include "Utility/LocalConfigurator.h"
#include "Utility/GLHelpers.h"
#include "Utility/CudaToolKit.h"

class Visualizer {
public:
    Visualizer() {}

    virtual ~Visualizer() {
    }

    static void SetParameters(Parameters &_params) {
        params = _params;
        input_size = params.InputSize;
    }

    void Setup();

    void Run() {
        glutMainLoop();
    }

    void PostProcessing() {
    }

    void UpdateWindowWithRGBAndDepth(void *rgb, void * depth);
    void UpdateWindowWithRGBAndDepthOnDevice(
            Image<uchar4, HostDeviceAllocator> & rgb,
            Image<float, HostDeviceAllocator> & depth,
            Image<float3, DeviceAllocator> & show,
            Image<float3, DeviceAllocator> & show2
            );
    bool Terminated();

protected:
    static void display();
    static void reshape(int width, int height);
    static void keys(unsigned char key, int x, int y);
    static void idle();
    void GLSetup();
    void TestSetup();

private:
    static Parameters params;
    static uint2 input_size;
    static bool terminated;

    static std::mutex rgb_mutex;
    static Image<float3, HostDeviceAllocator> m_vertex;
    static Image<float, HostDeviceAllocator> m_depth;
    static Image<uchar4, HostDeviceAllocator> m_rgb;

    static Image<float3, HostDeviceAllocator> m_show;
    static Image<float3, HostDeviceAllocator> m_show2;
};

