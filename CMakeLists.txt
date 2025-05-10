cmake_minimum_required(VERSION 3.8)
project("dragon.cpp")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD 11)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

option(LLAMA_ALL_WARNINGS            "llama: enable all compiler warnings"                   ON)
option(LLAMA_ALL_WARNINGS_3RD_PARTY  "llama: enable all compiler warnings in 3rd party libs" OFF)

if (APPLE)
    option(LLAMA_NO_ACCELERATE       "llama: disable Accelerate framework" OFF)
    option(LLAMA_NO_AVX              "llama: disable AVX" OFF)
    option(LLAMA_NO_AVX2             "llama: disable AVX2" OFF)
    option(LLAMA_NO_FMA              "llama: disable FMA" OFF)
endif()


if (APPLE AND NOT LLAMA_NO_ACCELERATE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if (ACCELERATE_FRAMEWORK)
        message(STATUS "Accelerate framework found")

        set(LLAMA_EXTRA_LIBS  ${LLAMA_EXTRA_LIBS}  ${ACCELERATE_FRAMEWORK})
        set(LLAMA_EXTRA_FLAGS ${LLAMA_EXTRA_FLAGS} -DDRAGON_USE_ACCELERATE)
    else()
        message(WARNING "Accelerate framework not found")
    endif()
endif()

if (LLAMA_ALL_WARNINGS)
    if (NOT MSVC)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} \
            -Wall                           \
            -Wextra                         \
            -Wpedantic                      \
            -Wshadow                        \
            -Wcast-qual                     \
            -Wstrict-prototypes             \
            -Wpointer-arith                 \
            -Wno-unused-function            \
        ")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} \
            -Wall                           \
            -Wextra                         \
            -Wpedantic                      \
            -Wcast-qual                     \
        ")
    else()
        # todo : msvc
    endif()
endif()

message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
else()
    message(STATUS "x86 detected")
    if (MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /arch:AVX2")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /arch:AVX2")
    else()
        if(NOT LLAMA_NO_AVX)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx")
        endif()
        if(NOT LLAMA_NO_AVX2)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx2")
        endif()
        if(NOT LLAMA_NO_FMA)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mfma")
        endif()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mf16c")
    endif()
endif()

# if (LLAMA_PERF)
#     set(LLAMA_EXTRA_FLAGS ${LLAMA_EXTRA_FLAGS} -DDRAGON_PERF)
# endif()

add_executable(llama
    main.cpp
    utils.cpp
    utils.h
    attention.cpp
    attention.h)

add_library(operators
        operators.c
        operators.h)

target_compile_definitions(operators PUBLIC ${LLAMA_EXTRA_FLAGS})
target_compile_definitions(llama PUBLIC ${LLAMA_EXTRA_FLAGS})


target_link_libraries(operators PRIVATE ${LLAMA_EXTRA_LIBS})
target_include_directories(operators PUBLIC .)
target_link_libraries(llama PRIVATE operators)
target_link_libraries(operators PRIVATE Threads::Threads)
