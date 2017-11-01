#!/bin/bash

filelist=`ls *.wav`
for file in $filelist
do
    output="16k_"$file
    sox $file -r 16000 $output
    rm $file
done
