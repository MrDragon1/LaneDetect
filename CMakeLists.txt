cmake_minimum_required(VERSION 3.17)
project(DIPFinal)

set(CMAKE_CXX_STANDARD 14)
set(OpenCV_DIR "G:/opencv/build")

find_package(OpenCV REQUIRED)

set(OpenCV_INCLUDE_DIRS "G:/opencv/build" "G:/Downloads/jsoncpp-master/jsoncpp-master/include")
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(DIPFinal main.cpp)

set(OpenCV_LIBS opencv_core opencv_imgproc opencv_highgui opencv_imgcodecs)

target_link_libraries(DIPFinal ${OpenCV_LIBS})
