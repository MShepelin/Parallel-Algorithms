cmake_minimum_required(VERSION 3.2)
project(parallel_rank LANGUAGES C CXX CUDA)
enable_language(CUDA)

include(GenerateExportHeader)

if(MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
endif()

find_package(CUDA REQUIRED)

add_library(glut STATIC IMPORTED GLOBAL)
set_target_properties(glut PROPERTIES IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/parallelrank/glut64.lib")
add_library(parallel_rank SHARED ./parallelrank/helpers.c ./parallelrank/kernel.cu)
target_link_libraries(parallel_rank glut)

target_compile_options(parallel_rank PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
  --use_fast_math
  #--relocatable-device-code=true
  -Xptxas 
  -O3
>)

include_directories(${CUDA_INCLUDE_DIRS})
set(THRUST_LIBRARIES ${CUDA_CUDA_LIBRARY})
target_link_libraries(parallel_rank ${THRUST_LIBRARIES})
