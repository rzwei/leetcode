cmake_minimum_required(VERSION 3.5)
project(leetcode)

set(CMAKE_CXX_STANDARD 11)
set(SOURCE_FILES  leetcode4.cpp)
SET(CMAKE_CXX_COMPILER clang++)
add_executable(leetcode ${SOURCE_FILES})