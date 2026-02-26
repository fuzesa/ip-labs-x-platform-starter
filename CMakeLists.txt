cmake_minimum_required(VERSION 4.0 FATAL_ERROR)
project(utcn_ip_starter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(${PROJECT_NAME}
    src/main.cc
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_LIST_DIR}/include
)

if(WIN32)
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/lib/windows/nfd.lib
    )
endif()

# On macOS, link required frameworks
if(APPLE)
    target_link_libraries(${PROJECT_NAME}
        PRIVATE
            ${CMAKE_CURRENT_LIST_DIR}/lib/macos/libnfd.a
            "-framework AppKit"
            "-framework UniformTypeIdentifiers"
    )
endif()
