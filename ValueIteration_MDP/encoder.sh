#! /bin/bash
if [ $# == 1 ]; then
    python encoder.py $1 1.0
else
    python encoder.py $1 $2
fi