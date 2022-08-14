cmake_minimum_required(VERSION 3.14)

project(OCR)

add_definitions(-std=c++17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Ofast -DNDEBUG")

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

set(TensorRT_DIR "" CACHE PATH "location of tensorrt")
set(CUDA_TOOLKIT_ROOT_DIR "" CACHE PATH "location of cuda")
set(OpenCV_DIR "" CACHE PATH "location of opencv")


find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
find_package(OpenCV 3 REQUIRED)
find_package(GFLAGS REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)

#message(${TensorRT_LIBRARIES})
#message(${TensorRT_INCLUDE_DIRS})
message(${OpenCV_VERSION})

file(GLOB SRC_FILE src/*.cpp)

add_executable(rec main.cpp ${SRC_FILE})

target_include_directories(rec PUBLIC ${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${TensorRT_INCLUDE_DIRS}
        ${TensorRT_DIR}/samples/common)
target_link_libraries(rec  ${OpenCV_LIBS} ${CUDA_LIBRARIES} ${TensorRT_LIBRARIES} ${GFLAGS_LIBRARIES})

add_definitions(-O2 -pthread)