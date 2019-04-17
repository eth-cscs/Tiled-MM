################
#  Build test  #
################
add_executable(multiply "multiply.cpp")
target_link_libraries(multiply tiled-mm options ${DEPENDENCIES})

add_executable(cublasXt-multiply "cublasXt-multiply.cpp")
target_link_libraries(cublasXt-multiply tiled-mm options ${DEPENDENCIES})
