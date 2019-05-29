cmake_minimum_required(VERSION 3.5)
project(leetcode)

set(CMAKE_CXX_STANDARD 17)
set(SOURCE_FILES leetcode1051.cpp)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
SET(CMAKE_CXX_COMPILER clang++)
add_executable(leetcode ${SOURCE_FILES})
