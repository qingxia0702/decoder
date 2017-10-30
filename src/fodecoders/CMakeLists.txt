cmake_minimum_required(VERSION 2.8.3)
project(cvte_decoder)

set(SUB_PROJECT_NAME "fodecoders")
#Define root paths
set(PROJECT_PATH "/Users/qing/SpeakIn/decoder")
set(KALDI_ROOT_PATH "/Users/qing/project/kaldi")

#Add head files for project(.h)
include_directories("${PROJECT_PATH}/include")
#include_directories("${PROJECT_PATH}/include/ATLAS/include")
#include_directories("${PROJECT_PATH}/include/fst")
#include_directories("${PROJECT_PATH}/include/CLAPACK")
include_directories("/usr/include")
include_directories("${KALDI_ROOT_PATH}/tools/openfst/include")
include_directories("${KALDI_ROOT_PATH}/tools/ATLAS/include")
include_directories("/usr/local/include")
#############################################################
#Be sure that include directories above should not overlap!
#############################################################

#Add lib files for project(.so)
link_directories("/usr/local/lib")
link_directories("/usr/lib")
link_directories("${PROJECT_PATH}/libs")
#Mac OS use Accelerate framework as atlas
IF (APPLE)
    message("LOG: find atlas for Mac")
    find_library(ATLAS Accelerate)
    if (NOT ATLAS)
            message(FATAL_ERROR "AudioUnit not found")
    endif()
    set(ATLAS_PATH ":")
ELSE()
    message("LOG: set atlas path for Unix")
    set(ATLAS_PATH ":/usr/lib/atlas-base")
ENDIF ()

#Set definitions and flags of compiler
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
IF (APPLE)
    message("LOG: Cmake definition for Mac")
    set(CMAKE_D -DKALDI_DOUBLEPRECISION=0 -DHAVE_EXECINFO_H=1 
        -DHAVE_CXXABI_H -DHAVE_CLAPACK -std=c++11)
ELSE ()
    message("LOG: Cmake definition for Unix")
    set(CMAKE_D -DKALDI_DOUBLEPRECISION=0 -DHAVE_EXECINFO_H=1 
        -DHAVE_CXXABI_H -DHAVE_ATLAS -std=c++11)
ENDIF ()
add_definitions(${CMAKE_D})
set(CMAKE_CXX_FLAGS "-rdynamic -g -o -Wno-sign-compare -Wno-unused-local-typedefs -Wno-deprecated-declarations -Winit-self -msse -msse2 -pthread -fPIC")
#set(CMAKE_CXX_FLAGS "-g -o -Wno-sign-compare -Wno-unused-local-typedefs -Wno-deprecated-declarations -Winit-self -msse -msse2 -pthread -fPIC")
#set(CMAKE_CXX_FLAGS "-g -o -msse -msse2 -pthread -fPIC")

#Definition of link libraries list in kaldi
set(KALDI_LIB   kaldi-online2 kaldi-ivector kaldi-nnet3 kaldi-nnet2
                kaldi-chain kaldi-cudamatrix kaldi-decoder kaldi-fstext
                kaldi-lat kaldi-hmm kaldi-feat kaldi-transform 
                kaldi-gmm kaldi-tree kaldi-util kaldi-matrix kaldi-base)
#Defintiont link libraries list system
IF (APPLE)
    message("LOG: system link for Mac")
    set(SYSTEM_LIB fst pthread dl m)
ELSE ()
    message("LOG: system link for Unix")
    set(SYSTEM_LIB atlas f77blas cblas lapack_atlas fst pthread dl m)
ENDIF ()


#Add executable file which to generate
#Add dynamic link library for executable file
add_executable(decoder decoder.cc)
target_link_libraries(decoder ${KALDI_LIB} ${SYSTEM_LIB})
add_executable(phone-decoder phone-decoder.cc)
target_link_libraries(phone-decoder ${KALDI_LIB} ${SYSTEM_LIB})
##############################################################
# Be sure that atlas f77blas cblas lapack_atlas are exist in
# lib directories, an be sure name of .so file should be:
#   libatlas.so
#   libf77blas.so
#   libcblas.so
#   liblapack_atlas.so
# If is no, you should new a link file named like above!
##############################################################

#Set install properties
IF (APPLE)
    message("Set targets properties on Mac OS")
    set_target_properties(decoder PROPERTIES 
                          INSTALL_RPATH "${PROJECT_PATH}/libs")
ELSE (UNIX)
    message("Set targets properties on unix")
    set_target_properties(decoder PROPERTIES 
                          INSTALL_RPATH "${PROJECT_PATH}/libs:${ATLAS_PATH}")
ENDIF ()
IF (APPLE)
    message("Set targets properties on Mac OS")
    set_target_properties(phone-decoder PROPERTIES 
                          INSTALL_RPATH "${PROJECT_PATH}/libs")
ELSE (UNIX)
    message("Set targets properties on unix")
    set_target_properties(phone-decoder PROPERTIES 
                          INSTALL_RPATH "${PROJECT_PATH}/libs:${ATLAS_PATH}")
ENDIF ()

install(TARGETS decoder phone-decoder
    RUNTIME DESTINATION ${PROJECT_PATH}/bin/${SUB_PROJECT_NAME})

