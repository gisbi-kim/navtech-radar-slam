#!/bin/bash
mkdir 2020_11_26
cd 2020_11_26
aws s3 sync boreas-2020-11-26-13-58 . --exclude '*' --include 'applanix/*' --include 'calib/*' --include 'lidar/*' --include 'radar/*' --include '*.html'
