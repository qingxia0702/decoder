#!/bin/bash
#Function: Init the decoder project for cvte asoutics model
#@para1: Directory of the kaldi project root 
#@para2: Directory of cvte/other asoutics model
#@para3: Directory of cvte/other HCLG.fst 
#   Usage: init_project.sh kaldi-root-path final.mdl HCLG.fst words.txt

if [ $# != 1 ]
then
    echo "ERROR: Please entry the right paramentes!"
    echo "Usage: init_project.sh kaldi-root-path final.mdl HCLG.fst words.txt"
    exit
fi


kaldi_root_path=$1
asoutics_modle=$2
fst_model=$3
words=$4

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

#filecheck include libs models
filecheck include libs

#cp $asoutics_modle models
#cp $fst_model models
#cp $words models 

#Copy .so files from cvte-decoder dependence to libs
cp $kaldi_root_path/src/lib/* libs 
cp -r $kaldi_root_path/tools/openfst/lib/* libs

# Copy .h files from kaldi to include
find $kaldi_root_path/src -name "*.h" >> head.list
while read line
do
    path=`echo ${line%/*}`
    sub=`echo ${path##*/}`
    if [ ! -d include/$sub ]
    then
        mkdir -p include/$sub
    fi
    cp $line include/$sub
done < head.list 
rm head.list
cp -r $kaldi_root_path/tools/openfst/include/* include
cp -r $kaldi_root_path/tools/ATLAS include 
cp -r $kaldi_root_path/tools/CLAPACK include 



