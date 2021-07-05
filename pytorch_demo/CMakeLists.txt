cmake_minimum_required(VERSION 3.4.1)
project(torchdemo)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++14")

set(libtorch_VERSION 1.8.0)
set(CMAKE_PREFIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/libtorch)
find_package(Torch ${libtorch_VERSION} EXACT QUIET CONFIG)
if(NOT Torch_FOUND)
  message(STATUS "libtorch ${libtorch_VERSION} - not found")
  message(STATUS "Fetching libtorch")
  include(FetchContent)
  FetchContent_Declare(
          libtorch
          URL https://download.pytorch.org/libtorch/cpu/libtorch-macos-${libtorch_VERSION}.zip
          SOURCE_DIR libtorch)
  FetchContent_GetProperties(libtorch)
  if(NOT libtorch_POPULATED)
    unset(FETCHCONTENT_QUIET CACHE)
    FetchContent_Populate(libtorch)
    list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR}/libtorch)
  endif()
  find_package(Torch ${libtorch_VERSION} EXACT CONFIG REQUIRED)
else()
  message(STATUS "libtorch ${libtorch_VERSION} - found")
endif()


add_executable(torchdemo "src/main.cpp")
include_directories(torchdemo PUBLIC
    ${TORCH_INCLUDE_DIRS}
)
target_link_libraries(torchdemo ${TORCH_LIBRARIES})