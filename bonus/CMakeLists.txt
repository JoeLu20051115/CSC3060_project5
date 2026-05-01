cmake_minimum_required(VERSION 3.12)
project(code_opt_bonus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -mfma -ffast-math -Wall -g -fno-omit-frame-pointer")

file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/../src/kernel/*.cpp"
)

add_library(kernels_bonus STATIC ${KERNEL_SOURCES})

target_include_directories(kernels_bonus
    PUBLIC
        "${CMAKE_SOURCE_DIR}/../include"
)

target_compile_options(kernels_bonus PRIVATE
    -O3 -mavx2 -mfma -ffast-math -Wall
)

add_executable(run_all_bonus "${CMAKE_SOURCE_DIR}/../src/main/run_all.cpp")
target_link_libraries(run_all_bonus PRIVATE kernels_bonus)

target_compile_options(run_all_bonus PRIVATE
    -O3 -mavx2 -mfma -ffast-math -Wall
)

add_executable(single_bench_bonus "${CMAKE_SOURCE_DIR}/../src/main/single_bench.cpp")
target_link_libraries(single_bench_bonus PRIVATE kernels_bonus)

target_compile_options(single_bench_bonus PRIVATE
    -O3 -mavx2 -mfma -ffast-math -Wall
)
