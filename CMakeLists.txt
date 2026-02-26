cmake_minimum_required(VERSION 4.0)
project(utcn_ip_starter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
# MacOS and Linux
if(UNIX)
    find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs)
endif()

add_executable(${PROJECT_NAME}
  src/main.cc
)

target_include_directories(${PROJECT_NAME}
  PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/include
)

add_library(nfd::nfd STATIC IMPORTED GLOBAL)

if(APPLE)
    set_target_properties(nfd::nfd PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/lib/macos/libnfd.a"
  )
    target_link_libraries(nfd::nfd INTERFACE
    "-framework AppKit"
    "-framework UniformTypeIdentifiers"
  )
elseif(WIN32)
    set_target_properties(nfd::nfd PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/lib/windows/nfd.lib"
  )
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set_target_properties(nfd::nfd PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/lib/linux/libnfd.a"
  )
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        nfd::nfd
        opencv_core
        opencv_imgproc
        opencv_highgui
        opencv_imgcodecs
)
