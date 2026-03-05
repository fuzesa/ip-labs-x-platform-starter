cmake_minimum_required(VERSION 4.0)
project(utcn_ip_starter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(APPLE)
  # CMAKE_OSX_SYSROOT=$(xcrun --show-sdk-path)
  set(CMAKE_OSX_SYSROOT "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX.sdk")
endif(APPLE)

# Find GTK3 for Linux (required by NFD)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GTK3 REQUIRED gtk+-3.0)
endif()

# Find OpenCV
if(UNIX)
    # MacOS and Linux should be installed via package manager
    find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs)
else()
    # Windows
    find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs PATHS "${CMAKE_CURRENT_LIST_DIR}/lib/windows/opencv")
endif()

add_definitions(-DASSETS_DIR="${CMAKE_SOURCE_DIR}/assets/")

add_executable(${PROJECT_NAME}
  src/file-util.cc
  src/image-util.cc
  src/terminal-util.cc
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
    target_link_libraries(nfd::nfd INTERFACE ${GTK3_LIBRARIES})
    target_include_directories(nfd::nfd INTERFACE ${GTK3_INCLUDE_DIRS})
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

if(WIN32)
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:opencv_world>
            $<TARGET_FILE_DIR:${PROJECT_NAME}>
    )
endif()