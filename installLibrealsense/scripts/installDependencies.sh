#!/bin/bash
# Script to install librealsense dependencies on Jetson
# author : nelsoonc - Mechanical Engineering 2017

green=`tput setaf 2`
reset=`tput sgr0`

echo "${green}Update and upgrade ubuntu packages${reset}"
sudo apt-get update
sudo apt-get upgrade

# Install dependencies used in the desired configuration
echo "${green}Installing build dependencies${reset}"
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    zip \
    libssl-dev \
    libusb-1.0.0-dev \
    pkg-config \
    libgtk-3-dev \
    python3-dev