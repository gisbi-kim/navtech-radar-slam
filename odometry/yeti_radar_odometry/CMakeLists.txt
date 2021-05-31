project(yeti)
cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -W -Wno-unused-parameter -O2")

find_package(OpenMP REQUIRED)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(OMP_NUM_THREADS 8)

find_package(OpenCV 3 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS
    system
    filesystem
)
find_package(PkgConfig)

find_package(PythonLibs REQUIRED)

find_package(PCL REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  rosbag
  std_msgs
  image_transport
  cv_bridge
  tf
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs
  DEPENDS EIGEN3 PCL 
  INCLUDE_DIRS include
)

include_directories(
	include
    third_party/matplotlib-cpp
	third_party/nanoflann
	${catkin_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIR}
)

get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
foreach(dir ${dirs})
  message(STATUS "dir='${dir}'")
endforeach()

add_library(features
  src/features.cpp
)

target_link_libraries(features
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	${Boost_INCLUDE_DIRS}
    Eigen3::Eigen
)

add_library(radar_utils
  src/radar_utils.cpp
)

target_link_libraries(radar_utils
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	${Boost_INCLUDE_DIRS}
    Eigen3::Eigen
)

add_library(association
  src/association.cpp
)

target_link_libraries(association
	${catkin_LIBRARIES}
	${Boost_INCLUDE_DIRS}
    Eigen3::Eigen
)

add_executable(visualization src/visualization.cpp)
target_link_libraries(visualization
    features
    radar_utils
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	${Boost_INCLUDE_DIRS}
)

add_executable(yetiOdom src/odometry.cpp)
target_link_libraries(yetiOdom
    features
    radar_utils
    association
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	${PCL_LIBRARIES}
)

add_library(matplotlibcpp INTERFACE)
target_include_directories(matplotlibcpp INTERFACE include)
target_include_directories(matplotlibcpp INTERFACE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(matplotlibcpp INTERFACE ${PYTHON_LIBRARIES})

add_executable(test_motiondistortion src/test_motiondistortion.cpp)
target_link_libraries(test_motiondistortion
    association
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	matplotlibcpp
)
target_compile_definitions(test_motiondistortion PRIVATE "-DWITHOUT_NUMPY")

add_executable(showcase_distortion src/showcase_distortion.cpp)
target_link_libraries(showcase_distortion
    features
    radar_utils
    association
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
	matplotlibcpp
)
target_compile_definitions(showcase_distortion PRIVATE "-DWITHOUT_NUMPY")

add_executable(localization src/localization.cpp)
target_link_libraries(localization
    features
    radar_utils
    association
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
)

add_executable(save_cartesians src/save_cartesians.cpp)
target_link_libraries(save_cartesians
    radar_utils
	${catkin_LIBRARIES}
	${OpenCV_LIBRARIES}
)
