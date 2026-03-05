cmake_minimum_required(VERSION 3.14)
project(fastmatch VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find dependencies
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# ONNX Runtime — user can set ONNXRUNTIME_ROOT or we try pkg-config / brew
if(NOT ONNXRUNTIME_ROOT)
    # Try common Homebrew path on macOS
    if(APPLE)
        set(ONNXRUNTIME_SEARCH_PATHS
            /opt/homebrew/opt/onnxruntime
            /usr/local/opt/onnxruntime
            /opt/homebrew
            /usr/local
        )
    else()
        set(ONNXRUNTIME_SEARCH_PATHS /usr/local /usr)
    endif()

    foreach(path ${ONNXRUNTIME_SEARCH_PATHS})
        if(EXISTS "${path}/include/onnxruntime/onnxruntime_cxx_api.h")
            set(ONNXRUNTIME_ROOT ${path})
            break()
        endif()
        if(EXISTS "${path}/include/onnxruntime_cxx_api.h")
            set(ONNXRUNTIME_ROOT ${path})
            break()
        endif()
    endforeach()
endif()

if(NOT ONNXRUNTIME_ROOT)
    message(FATAL_ERROR "ONNX Runtime not found. Set -DONNXRUNTIME_ROOT=/path/to/onnxruntime")
endif()

message(STATUS "ONNX Runtime root: ${ONNXRUNTIME_ROOT}")

# Find ONNX Runtime headers and library
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS ${ONNXRUNTIME_ROOT}/include ${ONNXRUNTIME_ROOT}/include/onnxruntime
    NO_DEFAULT_PATH
)
find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS ${ONNXRUNTIME_ROOT}/lib
    NO_DEFAULT_PATH
)

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "Could not find ONNX Runtime headers/library in ${ONNXRUNTIME_ROOT}")
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIBRARY}")

# Build executable
add_executable(fastmatch
    src/main.cpp
    src/object_detection.cpp
    src/feature_extraction.cpp
    src/helpers.cpp
)

target_include_directories(fastmatch PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${YAML_CPP_INCLUDE_DIR}
    src
)

target_link_libraries(fastmatch PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
    yaml-cpp::yaml-cpp
)

# Optimize for speed
target_compile_options(fastmatch PRIVATE -O3 -march=native)
