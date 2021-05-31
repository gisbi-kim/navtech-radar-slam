FROM ubuntu:xenial

RUN apt-get update && apt-get install -y \
    dirmngr \
    gnupg2 \
    wget \
    htop \
    vim \
    build-essential \
    gcc-5 \
    g++-5 \
    git

# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu xenial main" > /etc/apt/sources.list.d/ros1-latest.list

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

ENV ROS_DISTRO kinetic

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-kinetic-ros-core=1.3.2-0*

RUN apt-get install -y \
    ros-kinetic-eigen* \
    ros-kinetic-opencv* \
    python-catkin-tools

RUN echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc

env GIT_TOKEN # replace this your own git token!

RUN git clone https://$GIT_TOKEN@github.com/ethz-asl/libnabo.git
RUN cd libnabo && mkdir build && cd build && cmake .. && make -j32 && make install

RUN git clone https://$GIT_TOKEN@github.com/ethz-asl/libpointmatcher.git
RUN cd libpointmatcher && mkdir build && cd build && cmake .. && make -j32 && make install

RUN apt-get install -y libeigen3-dev

RUN mkdir steam_ws && cd steam_ws && git clone https://$GIT_TOKEN@github.com/utiasASRL/steam.git && \
    cd steam && git submodule update --init --remote && \
    cd deps/catkin && catkin config --extend /opt/ros/kinetic && catkin build && cd ../.. && \
    catkin build

RUN git clone https://$GIT_TOKEN@github.com/jbeder/yaml-cpp.git
RUN cd yaml-cpp && mkdir build && cd build && cmake .. && make -j32 && make install

RUN mkdir catkin_ws && cd catkin_ws && mkdir src && cd src && \
    git clone https://$GIT_TOKEN@github.com/keenan-burnett/yeti.git && \
    cd .. && catkin config --extend /steam_ws/devel/repo && \
    catkin build
