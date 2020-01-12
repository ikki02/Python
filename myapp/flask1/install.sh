#!/bin/bash

# install flask and Mecab on Ubuntu

echo 'export PATH="${HOME}/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
sudo apt update && sudo apt install python3-pip
python3 -m pip install --user flask mecab-python3

sudo apt install mecab libmecab-dev mecab-ipadic-utf8
sudo apt install swig

"
git clone https://github.com/neologd/mecab-ipadic-neologd.git
cd mecab-ipadic-neologd
sudo bin/install-mecab-ipadic-neologd
dicdir = /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd
"
