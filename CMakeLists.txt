# cmake needs this line
cmake_minimum_required(VERSION 3.10)

# Define project name
project(vits)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "v")
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

find_path(ONNX_RUNTIME_SESSION_INCLUDE_DIRS onnxruntime_cxx_api.h HINTS onnxruntime/include/)
find_library(ONNX_RUNTIME_LIB onnxruntime HINTS onnxruntime/lib)

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB_RECURSE SOURCES "${PROJECT_SOURCE_DIR}/main.cpp" "${PROJECT_SOURCE_DIR}/src/*.cpp" "${PROJECT_SOURCE_DIR}/src/*.c" "${PROJECT_SOURCE_DIR}/src/*.h" "${PROJECT_SOURCE_DIR}/src/*.hpp")

add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(vits PRIVATE ${ONNX_RUNTIME_SESSION_INCLUDE_DIRS} )
target_link_libraries(vits PRIVATE ${ONNX_RUNTIME_LIB})
