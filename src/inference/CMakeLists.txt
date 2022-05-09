cmake_minimum_required(VERSION 3.18)

project(Inference VERSION 1.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set (CMAKE_CXX_FLAGS "-O3 -pthread")
add_definitions(-w)

find_package(CUDAToolkit REQUIRED)
include(cmake/FindCudnn.cmake)

if(NOT ${CUDNN_FOUND})
    message(FATAL_ERROR "Not found CUDNN!")
endif()

if(NOT TRT_ROOT)
    message(FATAL_ERROR "Please set TRT_ROOT.")
endif()

# find tensorRT libs
list(APPEND PLUGINS "nvinfer")
list(APPEND PLUGINS "nvonnxparser")
list(APPEND PLUGINS "nvparsers")

foreach(libName ${PLUGINS})
    find_library(${libName}_lib NAMES ${libName} PATHS ${TRT_ROOT} PATH_SUFFIXES lib REQUIRED)
    list(APPEND TRT_LIBS "${${libName}_lib}")
endforeach()

set(TRT_INCLUDE_DIR ${TRT_ROOT}/include())

include_directories(${TRT_INCLUDE_DIR} ${CUDNN_INCLUDE_DIR} common)

file(GLOB_RECURSE files RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} common/*.cpp)

add_executable(sample Sample.cpp ${files})
target_link_libraries(sample PRIVATE CUDA::cudart ${CUDNN_LIBRARIES} ${TRT_LIBS} ${CMAKE_DL_LIBS})
