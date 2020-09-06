#! /bin/bash

if [ $# == 2 ]; then
    python decoder.py $1 $2 1.0
else
    python decoder.py $1 $2 $3
fi