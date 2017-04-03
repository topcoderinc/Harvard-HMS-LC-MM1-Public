#!/bin/bash
sudo apt-get install virtualenv
virtualenv -p /usr/bin/python3.5 $HOME/hms
source $HOME/hms/bin/activate
pip install -r ./requirements.txt
