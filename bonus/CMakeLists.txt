cmake_minimum_required(VERSION 3.12)
project(code_opt_bonus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 基础全局编译选项：启用 AVX-512 宽度偏好，强制开启 OpenMP 和极速数学
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mprefer-vector-width=512 -ffast-math -funroll-loops -fopenmp")

# 获取源文件
file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/../src/kernel/*.cpp"
)

file(GLOB BONUS_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/*.cpp"
)

# 构建静态库
add_library(kernels_bonus STATIC ${KERNEL_SOURCES} ${BONUS_SOURCES})

# 恢复正确的头文件包含路径
target_include_directories(kernels_bonus
    PUBLIC
        "${CMAKE_SOURCE_DIR}/../include"
        "${CMAKE_SOURCE_DIR}"
)

# 为 kernel 强行绑定所有极限优化指令
target_compile_options(kernels_bonus PRIVATE
    -O3 -march=native -mprefer-vector-width=512 -ffast-math -funroll-loops
)

# ---------------------------------------------------------
# 寻找并链接 OpenMP
# ---------------------------------------------------------
find_package(OpenMP REQUIRED)
target_link_libraries(kernels_bonus PUBLIC OpenMP::OpenMP_CXX)

# ---------------------------------------------------------
# 寻找并链接 OpenBLAS
# ---------------------------------------------------------
find_library(OPENBLAS_LIB openblas)
find_library(BLAS_LIB blas)

if(OPENBLAS_LIB)
    message(STATUS "Found OpenBLAS: ${OPENBLAS_LIB}")
    set(BLAS_LIBS ${OPENBLAS_LIB})
elseif(BLAS_LIB)
    message(STATUS "Found BLAS: ${BLAS_LIB}")
    set(BLAS_LIBS ${BLAS_LIB})
endif()

if(BLAS_LIBS)
    target_compile_definitions(kernels_bonus PRIVATE BONUS_USE_BLAS)
    target_link_libraries(kernels_bonus PRIVATE ${BLAS_LIBS})
else()
    message(WARNING "OpenBLAS/BLAS not found; bonus will compile without BLAS acceleration.")
endif()

# ---------------------------------------------------------
# 编译可执行文件 (恢复为你原本正确的路径！)
# ---------------------------------------------------------
add_executable(run_all_bonus "${CMAKE_SOURCE_DIR}/../src/main/run_all.cpp")
# 必须把 OpenMP 链接到可执行文件，否则多线程可能无法拉起
target_link_libraries(run_all_bonus PRIVATE kernels_bonus OpenMP::OpenMP_CXX)
target_compile_options(run_all_bonus PRIVATE
    -O3 -march=native -mprefer-vector-width=512 -ffast-math
)

add_executable(single_bench_bonus "${CMAKE_SOURCE_DIR}/../src/main/single_bench.cpp")
target_link_libraries(single_bench_bonus PRIVATE kernels_bonus OpenMP::OpenMP_CXX)
target_compile_options(single_bench_bonus PRIVATE
    -O3 -march=native -mprefer-vector-width=512 -ffast-math
)