#!/bin/bash

if [ $# != 1 ]
then
    echo "ERROR: Please entry the right paramentes!"
    echo "Usage: init_project.sh kaldi-root-path final.mdl HCLG.fst words.txt"
    exit
fi

data_path=$1

filecheck(){
    for i in $@
    do
        if [ -d $i ]
        then
            rm -rf $i
        fi
        mkdir $i
    done
}
wava_data_list=`ls $data_path/*.wav`
for file in $wava_data_list
do
    output_path=${file%.*}
    mkdir -p $output_path
    # Do silence split option,
    # The splited file save in dir which named by wav file
    # sil_splite Usage:
    #   ./sil_splite final.mdl HCLG.fst *.wav words.txt $num_min_slience $wave_out_put_path
    #   num_mini_slience mean the mininize frame of silence you want to split";
    ./sil_splite ../../models/final.mdl ../../models/HCLG.fst.digiter $file \
        ../../models/words.txt 15 $output_path
done


