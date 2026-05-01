cmake_minimum_required(VERSION 3.12)
project(code_opt_bonus LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 基础全局编译选项：O3、AVX2、FMA(融合乘加)、本机架构优化、极速数学、循环展开
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx2 -mfma -march=native -mtune=native -ffast-math -funroll-loops -Wall")

# 获取源文件
file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/../src/kernel/*.cpp"
)

file(GLOB BONUS_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/*.cpp"
)

# 构建静态库
add_library(kernels_bonus STATIC ${KERNEL_SOURCES} ${BONUS_SOURCES})

target_include_directories(kernels_bonus
    PUBLIC
        "${CMAKE_SOURCE_DIR}/../include"
        "${CMAKE_SOURCE_DIR}"
)

# 为 kernel 强行绑定所有极限优化指令
target_compile_options(kernels_bonus PRIVATE
    -O3 -mavx2 -mfma -march=native -mtune=native -ffast-math -funroll-loops -Wall
)

# ---------------------------------------------------------
# 终极杀器 1：寻找并链接 OpenMP (多线程并发榨干 CPU)
# ---------------------------------------------------------
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "🔥 Found OpenMP: Multi-threading enabled!")
    # 现代 CMake 只需要 link 这个 target，就会自动处理 -fopenmp 等 flag
    target_link_libraries(kernels_bonus PRIVATE OpenMP::OpenMP_CXX)
else()
    message(WARNING "⚠️ OpenMP not found. Code will run on a single thread.")
endif()

# ---------------------------------------------------------
# 终极杀器 2：寻找并链接 OpenBLAS (调用顶尖数学库)
# ---------------------------------------------------------
find_library(OPENBLAS_LIB openblas)
find_library(BLAS_LIB blas)

if(OPENBLAS_LIB)
    message(STATUS "🚀 Found OpenBLAS: ${OPENBLAS_LIB}")
    set(BLAS_LIBS ${OPENBLAS_LIB})
elseif(BLAS_LIB)
    message(STATUS "🚀 Found BLAS: ${BLAS_LIB}")
    set(BLAS_LIBS ${BLAS_LIB})
endif()

if(BLAS_LIBS)
    target_compile_definitions(kernels_bonus PRIVATE BONUS_USE_BLAS)
    target_link_libraries(kernels_bonus PRIVATE ${BLAS_LIBS})
else()
    message(WARNING "⚠️ OpenBLAS/BLAS not found; bonus will compile without BLAS acceleration.")
endif()

# ---------------------------------------------------------
# 编译可执行文件
# ---------------------------------------------------------
add_executable(run_all_bonus "${CMAKE_SOURCE_DIR}/../src/main/run_all.cpp")
target_link_libraries(run_all_bonus PRIVATE kernels_bonus)
if(OpenMP_CXX_FOUND)
    target_link_libraries(run_all_bonus PRIVATE OpenMP::OpenMP_CXX)
endif()
target_compile_options(run_all_bonus PRIVATE
    -O3 -mavx2 -mfma -march=native -mtune=native -ffast-math -Wall
)

add_executable(single_bench_bonus "${CMAKE_SOURCE_DIR}/../src/main/single_bench.cpp")
target_link_libraries(single_bench_bonus PRIVATE kernels_bonus)
if(OpenMP_CXX_FOUND)
    target_link_libraries(single_bench_bonus PRIVATE OpenMP::OpenMP_CXX)
endif()
target_compile_options(single_bench_bonus PRIVATE
    -O3 -mavx2 -mfma -march=native -mtune=native -ffast-math -Wall
)