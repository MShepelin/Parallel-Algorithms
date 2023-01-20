cmake_minimum_required(VERSION 3.2)
project(parallel_rank LANGUAGES C CXX CUDA)
enable_language(CUDA)

include(GenerateExportHeader)

if(MSVC)
    set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS TRUE)
    set(BUILD_SHARED_LIBS TRUE)
endif()

add_library(glut STATIC IMPORTED GLOBAL)
set_target_properties(glut PROPERTIES IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/parallelrank/glut64.lib")
add_library(parallel_rank SHARED ./parallelrank/main.c ./parallelrank/kernel.cu)
target_link_libraries(parallel_rank glut)
