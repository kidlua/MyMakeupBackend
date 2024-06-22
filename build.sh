#!/bin/bash

# Ensure pip is updated
pip install --upgrade pip

# Install distutils
pip install setuptools

# Install the rest of the requirements
pip install -r requirements.txt
