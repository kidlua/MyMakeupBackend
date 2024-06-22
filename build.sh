#!/bin/bash

apt-get update
apt-get install -y python3-distutils
apt-get install -y python3-apt
pip3 install --upgrade pip
pip3 install -r requirements.txt
