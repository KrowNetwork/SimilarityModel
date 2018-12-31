#!/bin/bash
killall screen 

serving_port="9000"

git pull

screen -dm -S flask_server python serve.py $1
screen -dm -S tensorflow_server tensorflow_model_server --model_base_path=/home/tuckers_krow_network/SimilarityModel/models/ --rest_api_port=9000 --model_name=$1
