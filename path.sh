#!/bin/bash

eval "$(ssh-agent -s)"
ssh-add ~/.ssh/github

. ~/workspace/activate_python.sh

export HF_HOME=~/workspace/hub
