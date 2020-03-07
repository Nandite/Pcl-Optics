cmake_minimum_required(VERSION 3.15)
project(Pcl_Optics)

add_compile_options(-Wall -Wextra -O2)
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
        include/Optics.hpp main.cpp)

target_link_libraries(Pcl_Optics
        ${PCL_LIBRARIES})