#!/bin/bash

if [ $# != 1 ]
then
    echo "ERROR: Please entry the right paramentes!"
    echo "Usage: ./split.sh data-path"
    exit
fi

data_path=$1

sil_num=5


filecheck(){
    for i in $@
    do
        if [ -f $i ]
        then
            rm  $i
        fi
        touch $i
    done
}
wava_data_list=`ls $data_path/*.wav`

filecheck   splite.scp wav.scp
#filecheck wav.scp
#Generate the split.scp for call batch sil_splite
for file in $wava_data_list
do
    output_path=${file%.*}
    mkdir -p $output_path
    echo $file" "$output_path >> splite.scp
done
echo "======================================================"
echo "                 Begin to split                  "
echo "======================================================"
./batch_sil_splite ../../models/final.mdl ../../models/HCLG.fst.digiter  ark,t:splite.scp\
    ../../models/words.txt $sil_num
echo "======================================================"
echo "                 Split Done!!!!                  "
echo "======================================================"

#Find all splited directorys
id=1
splited_dir_list=`find $data_path -type d`
for dir in $splited_dir_list
do
    if [ $dir!=$data_path ] 
    then
        files_in_dir=`ls $dir/*.wav`
        for file in $files_in_dir
        do
            echo $file" "$id >> wav.scp 
            let id++
        done
    fi
done
echo "======================================================"
echo "                 Begin to label                  "
echo "======================================================"
./batch_decoder ../../models/final.mdl ../../models/HCLG.fst.digiter  ark,t:wav.scp ../../models/words.txt
echo "======================================================"
echo "                 Label Done!!!!                  "
echo "======================================================"









