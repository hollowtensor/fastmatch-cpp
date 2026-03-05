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

# Optional FAISS support (ON by default if found)
option(USE_FAISS "Use FAISS for fast similarity search" ON)

set(FAISS_FOUND FALSE)
if(USE_FAISS)
    if(APPLE)
        set(FAISS_SEARCH_PATHS /opt/homebrew/opt/faiss /usr/local/opt/faiss /opt/homebrew /usr/local)
    else()
        set(FAISS_SEARCH_PATHS /usr/local /usr)
    endif()

    find_path(FAISS_INCLUDE_DIR
        NAMES faiss/IndexFlat.h
        PATHS ${FAISS_SEARCH_PATHS}
        PATH_SUFFIXES include
    )
    find_library(FAISS_LIBRARY
        NAMES faiss
        PATHS ${FAISS_SEARCH_PATHS}
        PATH_SUFFIXES lib
    )

    if(FAISS_INCLUDE_DIR AND FAISS_LIBRARY)
        set(FAISS_FOUND TRUE)
        message(STATUS "FAISS found: ${FAISS_LIBRARY}")
    else()
        message(STATUS "FAISS not found — falling back to linear search")
    endif()
endif()

# Face detection & recognition demo
add_executable(face_demo
    src/faceid/face_demo.cpp
    src/faceid/scrfd.cpp
    src/faceid/arcface.cpp
    src/faceid/face_db.cpp
)

target_include_directories(face_demo PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
    src/faceid
)

target_link_libraries(face_demo PRIVATE
    ${OpenCV_LIBS}
    ${ONNXRUNTIME_LIBRARY}
)

if(FAISS_FOUND)
    target_compile_definitions(face_demo PRIVATE USE_FAISS)
    target_include_directories(face_demo PRIVATE ${FAISS_INCLUDE_DIR})
    target_link_libraries(face_demo PRIVATE ${FAISS_LIBRARY})
endif()

target_compile_options(face_demo PRIVATE -O3 -march=native)
