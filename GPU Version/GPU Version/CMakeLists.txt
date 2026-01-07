cmake_minimum_required(VERSION 3.10)

project(ElectronsProtons LANGUAGES CXX C CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_definitions(-DWIN32_LEAN_AND_MEAN -DNOMINMAX)

add_definitions(-D_CRT_SECURE_NO_WARNINGS)

include_directories(${CMAKE_SOURCE_DIR}/dependencies/include)
include_directories(${CMAKE_SOURCE_DIR})

link_directories(${CMAKE_SOURCE_DIR}/dependencies/libs)

set(SOURCES
    main.cpp
    kernel.cu
    workspace.cu
    glad.c
)

add_executable(ElectronsProtons ${SOURCES})

target_link_libraries(ElectronsProtons
    PRIVATE
    glfw3
    opengl32
    cudart
)

add_custom_command(TARGET ElectronsProtons POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${CMAKE_SOURCE_DIR}/dependencies/lib/glfw3.dll"
    $<TARGET_FILE_DIR:ElectronsProtons>
    COMMENT "Copying glfw3.dll to the folder with .exe file"
    VERBATIM
)