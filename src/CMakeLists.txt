#add_executable(testconfig TestConfigurator.cpp)

set(KinectFusionSrcs
        KinectFusion.h
        KinectFusion.cu
        Visualizer.cc
        Volume.cu
        helpers.cu
        Utility/LocalConfigurator.cu
        )

cuda_add_library(KinectFusion STATIC
                    ${KinectFusionSrcs}
                    OPTIONS
                    -O3
                    )

add_executable(testdisplay test.cc)
target_link_libraries(testdisplay 
                        KinectFusion
                        ${GLUT_LIBRARIES}
                        ${OPENGL_LIBRARIES}
                        ${OpenCV_LIBRARIES}
                        )
