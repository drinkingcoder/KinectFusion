#include "Visualizer.h"

Parameters Visualizer::params;
uint2 Visualizer::input_size;
bool Visualizer::terminated;

std::mutex Visualizer::rgb_mutex;
Image<float3, HostDeviceAllocator> Visualizer::m_vertex;
Image<float, HostDeviceAllocator> Visualizer::m_depth;
Image<uchar4, HostDeviceAllocator> Visualizer::m_rgb;

using namespace std;
namespace {
    void specials(int key, int x, int y) {
        switch (key) {
        }
        glutPostRedisplay();
    }

}

void Visualizer::display() {
    lock_guard<mutex> guard(rgb_mutex);
    cudaDeviceSynchronize();

    glRasterPos2i(0, 0);
    glDrawPixels(m_rgb);
    glRasterPos2i(params.InputSize.x, 0);
    glDrawPixels(m_depth);

    glutSwapBuffers();
}

void Visualizer::reshape(int width, int height) {
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glViewport(0, 0, width, height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glColor3f(1.0f, 1.0f, 1.0f);
    glRasterPos2f(-1, 1);
    glOrtho(-0.375, width - 0.375, height - 0.375, -0.375, -1, 1);
    glPixelZoom(1, -1);
}

void Visualizer::keys(unsigned char key, int x, int y) {
    switch (key) {
        case 'q': {
                      terminated = true;
                      exit(0);
                      break;
                  }
    }
    glutPostRedisplay();
}

void Visualizer::idle() {
    glutPostRedisplay();
}

void Visualizer::GLSetup() {
    glutInit(&(params.argc), params.argv);
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
    glutInitWindowSize(params.InputSize.x * 3, params.InputSize.y * 3);
    glutCreateWindow("KinectFusion Test");

    glutDisplayFunc(Visualizer::display);
    glutKeyboardFunc(keys);
    glutSpecialFunc(specials);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);
}

void Visualizer::TestSetup() {
}

void Visualizer::Setup() {
    GLSetup();
    m_rgb.Allocate(input_size);
    m_depth.Allocate(input_size);

    terminated = false;
}

bool Visualizer::Terminated() {
    return terminated;
}

void Visualizer::UpdateWindowWithRGBAndDepth(void * rgb, void * depth) {
    lock_guard<mutex> guard(rgb_mutex);
    CUDASafeCall(
            cudaMemcpy(
                m_rgb.GetDevicePtr(),
                rgb,
                input_size.x * input_size.y * 4,
                cudaMemcpyHostToDevice
                )
            );
    CUDASafeCall(
            cudaMemcpy(
                m_depth.GetDevicePtr(),
                depth,
                input_size.x * input_size.y * sizeof(float),
                cudaMemcpyHostToDevice
                )
            );
}

void Visualizer::UpdateWindowWithRGBAndDepthOnDevice(
        Image<uchar4, HostDeviceAllocator> &rgb,
        Image<float, HostDeviceAllocator> &depth
        ) {
    lock_guard<mutex> guard(rgb_mutex);
    m_depth = depth;
    m_rgb = rgb;
}
