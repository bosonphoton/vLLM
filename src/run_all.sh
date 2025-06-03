#!/bin/bas

mkdir -p data outputs plots
./run_data.sh
./run_verify.sh
./run_train.sh
./run_eval.sh
