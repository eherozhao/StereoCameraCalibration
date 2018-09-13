cmake_minimum_required(VERSION 2.6)
project(go)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#include_directories("usr/local/include/opencv")
include_directories("/home/hao/Documents/test/go")
aux_source_directory(. DIR_SRCS)

include_directories(${OpenCV_DIRS})
find_package( OpenCV REQUIRED )
add_executable(go ${DIR_SRCS})  


# aux_source_directory(. DIR_SRCS)
# add_executable(go ${DIR_SRCS})  
# message(${DIR_SRCS})

install(TARGETS go RUNTIME DESTINATION bin)
target_link_libraries( go ${OpenCV_LIBS} )
