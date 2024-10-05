#!/bin/bash


venv_type=conda

. tools/parse_options.sh


###############################
###   Step 1 Setup python   ###
###############################

# Create virtual environment if not exists
if [ ! -e ./activate_python.sh ]; then
  if [ ${venv_type} = "conda" ]; then
    . tools/setup_anaconda.sh miniconda ondev_dsu 3.10
  elif [ ${venv_type} = "venv" ]; then
    . tools/setup_venv.sh $(which python)
  fi
fi


#####################################
###   Step 2 Install dependency   ###
#####################################

# Activate python and install dependencies
. activate_python.sh
if [ ${venv_type} = "conda" ]; then
  . tools/install_torch.sh true 2.4.0 12.4
elif [ ${venv_type} = "venv" ]; then
  . tools/install_torch.sh false 2.4.0 12.4
fi

# install s3prl
(
  git clone https://github.com/s3prl/s3prl.git
  cd s3prl
  pip install -e .
)

# install espnet

(
  git clone https://github.com/espnet/espnet.git
  cd espnet
  pip install -e .
)

pip install -r tools/requirements.txt

