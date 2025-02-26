cmake_minimum_required(VERSION 3.26 FATAL_ERROR) # USE ANYTHING LESS THAN 3.27!!!!
project(ip-labs-x-platform-starter
        VERSION 1.0.0
        LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)

add_definitions(-DASSETS_DIR="${CMAKE_SOURCE_DIR}/assets/")

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/3rdparty/nfde")

add_executable(${PROJECT_NAME}
        src/util/file-util.cpp
        src/util/image-util.cpp
        src/util/terminal-util.cpp
        src/OpenCVApplication.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/include/util)

target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS} nfd)
