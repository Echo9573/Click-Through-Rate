cmake_minimum_required(VERSION 3.27)
project(CLionProjects)

set(CMAKE_CXX_STANDARD 17)

add_executable(CLionProjects main.cpp
        ListNode.h
        ListNode.cpp
        TreeNode.h
        TreeNode.cpp)
