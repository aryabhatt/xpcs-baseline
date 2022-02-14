
cmake_minimum_required(VERSION 3.8)

project(MDScatter CXX)
set(MDScatter_VERSION_MAJOR 1)
set(MDScatter_VERSION_MINOR 0)


# C++11 Stuff
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SRC dft.cpp)

# python d
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# pybind11
find_package(pybind11 REQUIRED)
set(LIBS pybind11::module)

find_package(OpenMP)
if(OpenMP_FOUND)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif(OpenMP_FOUND)


include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
	add_definitions("-DHAVE_CUDA")
	include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
	add_library(cudadft SHARED gpu_dft.cu)
	set_target_properties(cudadft PROPERTIES CUDA_SEPARABLE_COMPILATION ON) 
	set(LIBS ${LIBS} cudadft ${CUDA_CUDART_LIBRARY})
else()
	message(STATUS "No CUDA Support")
endif()

add_library(mdscatter MODULE ${SRC})
target_link_libraries(mdscatter PRIVATE ${LIBS})
set_target_properties(mdscatter PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                         SUFFIX "${PYTHON_MODULE_EXTENSION}")
                                        
