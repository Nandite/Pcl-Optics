cmake_minimum_required(VERSION 3.15)
project(Pcl_Optics)

add_compile_options(-Wall -Wextra -pedantic)
set(DEBUG_FLAGS "-g -ggdb -O0")
set(RELEASE_FLAGS "-03")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# PCL Package ##############################################################
find_package(PCL 1.9 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
############################################################################

include_directories(include)

add_executable(Pcl_Optics
        include/Optics.hpp
        optics_run.cpp)
target_link_libraries(Pcl_Optics
        ${PCL_LIBRARIES})

add_executable(Pcl_Optics_With_Indices
        include/Optics.hpp
        optics_indices_run.cpp)
target_link_libraries(Pcl_Optics_With_Indices
        ${PCL_LIBRARIES})