# TRT 
# requirement of cmake version
cmake_minimum_required(VERSION 3.0)
# project name
PROJECT(Repeater CXX)
set(CMAKE_CXX_STANDARD 14)

# set the directory of executable files
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)  # 输出文件夹路径
set(CMAKE_BUILD_TYPE "Release")  # 编译类型  Release Debug


# find required opencv
find_package(OpenCV REQUIRED)   # 使用find_package获取opencv的相关信息,如lib、include、version等
# find_package(glog REQUIRED)

# set(LIBRARIES_out  nvonnxparser nvinfer nvinfer_plugin ) 


# cuda
set(CUDA_ROOT_PATH /usr/local/cuda-11.1/) # 根目录
set(CUDA_INCLUDE_DIRS ${CUDA_ROOT_PATH}/include/)   # 头文件夹
set(CUDA_LIB ${CUDA_ROOT_PATH}lib64/libcudart.so)  # 依赖动态库
set(CUDA_NVCC_FLAGS -arch=compute_86;)
# name of executable file and path of source file
project(check_dimension CUDA) #指明为cuda文件
add_executable(Repeater src/repeater.cu)
set(CMAKE_CXX_FLAGS "-Wno-error=deprecated-declarations -Wno-deprecated-declarations ")
# directory of opencv headers
include_directories(${OpenCV_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS}) # 头文件夹
# directory of opencv library
link_directories(${OpenCV_LIBRARY_DIRS})    # 指定第三方库所在路径,该路径添加到第三方库搜索路径中
# opencv libraries
target_link_libraries(Repeater ${OpenCV_LIBS} ${TENSORRT_LIB} ${CUDA_LIB} ${glog_LIB})  # 相对路径

#导入一些库文件
target_link_libraries(${target_name}Repeater  ${CUDA_cublas_LIBRARY} ${CUDA_nppisu_LIBRARY} ${CUDA_nppif_LIBRARY} ${CUDA_nppif_LIBRARY})
# target_link_libraries( ${target_name}_test  ${CUDA_cublas_LIBRARY} ${CUDA_nppisu_LIBRARY} ${CUDA_nppif_LIBRARY} ${CUDA_nppif_LIBRARY})