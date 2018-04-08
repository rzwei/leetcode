cmake_minimum_required(VERSION 3.5)
project(leetcode)

set(CMAKE_CXX_STANDARD 17)
set(SOURCE_FILES  leetcode5.cpp)
SET(CMAKE_CXX_COMPILER clang++)
add_executable(leetcode ${SOURCE_FILES})