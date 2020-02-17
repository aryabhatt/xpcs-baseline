
cmake_minimum_required(VERSION 3.8)

project(MDScatter LANGUAGES CXX CUDA)
set(MDScatter_VERSION_MAJOR 1)
set(MDScatter_VERSION_MINOR 2)


# C++11 Stuff
set(CMAKE_CXX_STANDARD 11)
set(PYBIND11_CPP_STANDARD -std=c++11)

set(SRC dft.cpp pydft.cpp)

# cuda
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# python d
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(OpenMP)
if(OpenMP_FOUND)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OpenMP_FOUND)

find_package(pybind11)

add_library(dummycu STATIC gpu_dft.cu)
set_target_properties(dummycu PROPERTIES
                      POSITION_INDEPENDENT_CODE ON
                      CUDA_VISIBILITY_PRESET "hidden"
                     )

add_library(mdscatter MODULE ${SRC})
set_target_properties(mdscatter PROPERTIES
                      CXX_VISIBILITY_PRESET "hidden"
                      PREFIX "${PYTHON_MODULE_PREFIX}"
                      SUFFIX "${PYTHON_MODULE_EXTENSION}")
                                        
target_link_libraries(mdscatter PRIVATE dummycu)
target_link_libraries(mdscatter PRIVATE pybind11::module cudart)
