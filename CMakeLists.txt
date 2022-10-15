cmake_minimum_required(VERSION 3.0.0)
project(PnPSolver VERSION 0.1.0 LANGUAGES CXX)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    # Let's ensure -std=c++xx instead of -std=g++xx
    set(CMAKE_CXX_EXTENSIONS OFF)
    
    # Let's nicely support folders in IDE's
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

    include(CTest) # note: this adds a BUILD_TESTING which defaults to ON
endif()

# Release by default
# Turn on Debug with "-DCMAKE_BUILD_TYPE=Debug"
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Allow the developer to select if Dynamic or Static libraries are built
option (BUILD_SHARED_LIBS "Build Shared Libraries" ON)
set (LIB_TYPE STATIC)
if (BUILD_SHARED_LIBS)
  set (LIB_TYPE SHARED)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

# CMake module
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(Eigen3 REQUIRED)

add_library(PnPSolver ${LIB_TYPE}
  ${CMAKE_CURRENT_SOURCE_DIR}/include/pnp/cpnp.h
  ${CMAKE_CURRENT_SOURCE_DIR}/include/pnp/ops.h
  ${CMAKE_CURRENT_SOURCE_DIR}/src/cpnp.cpp)

target_include_directories(PnPSolver PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)

# Install
install(
    TARGETS PnPSolver
    EXPORT PnPSolverTargets
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    INCLUDES DESTINATION include
)

# copy public include files
install(FILES
        include/pnp/cpnp.h
        include/pnp/ops.h
        DESTINATION include/pnp)

# create export file which can be imported by other cmake projects
install(EXPORT PnPSolverTargets
        NAMESPACE pnp::
        DESTINATION lib/cmake/PnPSolver)

# generate the config file
include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/PnPSolverConfig.cmake
  INSTALL_DESTINATION lib/cmake/PnPSolver
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO)

write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/PnPSolverConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY AnyNewerVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/PnPSolverConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/PnPSolverConfigVersion.cmake
  DESTINATION lib/cmake/PnPSolver)

# Test
option(BUILD_TEST "Build test." ON)
if(BUILD_TEST)
    enable_testing()
    # Unit tests
    add_executable(test_pnp test/test_pnp.cpp)
    target_link_libraries(test_pnp PRIVATE PnPSolver)
endif()