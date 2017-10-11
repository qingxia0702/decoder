#README.md
[toc]
##1. Dependences
&emsp;`1. kaldi:` Make sure kaldi is installed and compiled!
&emsp;`2. cmake`
##2. Compile
###2.1 Initialization
`init_project.sh`
&emsp;To init the workspace of cvte decoder, Its main job is copy link libraries and head files from kaldi.
```powershell
#Usage:
	./init_project.sh kaldi-root-path final.mdl-path HCLG.fst-path words.txt-path
```
###2.2 Compile and install
```powershell
	mkdir build && cd build
	cmake ..
	make && make install
```
&emsp; The main CmakeLists.txt is `src/CMakeLists.txt`, There is some directory path need to modify. Before run `cmake ..`, you should modify  `PROJECT_PATH` and  `KALDI_ROOT_PATH` to you own directory path. Targets(***cvte-decoder and oneline-cvte-decoder***) is installed to `PROJECT_PATH/bin`
##3. Usage
```powershell
#Usage:
	./cvte-decoder final.mdl HCLG.fst 2017_03_07_16.57.40_2562.wav words.txt
	./online-cvte-decoder final.mdl HCLG.fst 2017_03_07_16.57.40_2562.wav words.txt
```
