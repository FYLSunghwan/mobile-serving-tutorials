cmake_minimum_required(VERSION 3.4.1)
project(tflitedemo)

set (TFLITE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/MNN)
set (TFLITE_LIB_PATH "${TFLITE_PATH}/shared/libMNN.dylib")
set (TFLITE_INCLUDE_PATH "${TFLITE_PATH}/include")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14")

add_library(lib_tflite SHARED IMPORTED)
set_target_properties(lib_tflite PROPERTIES IMPORTED_LOCATION
    ${TFLITE_LIB_PATH}
)

add_executable(tflitedemo "src/main.cpp")
include_directories(tflitedemo PUBLIC
    ${TFLITE_INCLUDE_PATH}
)
target_link_libraries(tflitedemo lib_tflite)