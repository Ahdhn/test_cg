
cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

#cmake_policy(SET CMP0077 OLD)

set(PROJECT_NAME "test_cg")
project(${PROJECT_NAME} LANGUAGES CXX CUDA)  

#default build type is Release
if (CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE Release)
endif ()

SET(CUDA_SEPARABLE_COMPILATION ON)

########## Libraries/flags Starts Here ######################
find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -lineinfo; -std=c++11; -expt-extended-lambda; -O3; -use_fast_math; -rdc=true)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode=arch=compute_70,code=sm_70) #for TITAN V
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -Wall -std=c++11")
########## Libraries/flags Ends Here ######################


########## inc/libs/exe/features Starts Here ######################
set(CMAKE_INCLUDE_CURRENT_DIR ON)
CUDA_ADD_EXECUTABLE(${PROJECT_NAME} test_cg.cu)

##add features to the library -> require it to compile with cuda c++11
target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)

##enable position-indep code 
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE  ON)

##enable the separate compilation on both device and host sides 
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} cuda)
########## inc/libs/exe/features Ends Here ######################