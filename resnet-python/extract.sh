#! /bin/bash

lib_dir="./lib"
model_dir="./resnet"
rm -rf $model_dir"/__pycache__"
require="./requirements.txt"
zip_name="resnet_deploy.zip"
zip -r $zip_name $model_dir $lib_dir $require -x */__pycache__/*
echo "successfully extract deploy packages to $zip_name"
