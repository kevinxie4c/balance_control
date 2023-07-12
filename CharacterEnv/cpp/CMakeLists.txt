project(balance_control)
cmake_minimum_required(VERSION 3.16)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

message("CCCDLFLAGS: ${CCCDLFLAGS}")
#set(CMAKE_CXX_FLAGS "-std=c++14 ${CCCDLFLAGS}")
set(CMAKE_CXX_FLAGS "${CCCDLFLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG")

add_library(CharacterEnv STATIC src/CharacterEnv.cpp src/SimCharacter.cpp src/MathUtil.cpp src/IOUtil.cpp src/ParallelEnv.cpp)
#target_link_libraries(CharacterEnv stdc++ dart assimp boost_system)
