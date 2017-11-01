**README.md**

**1. Dependences**

`1. kaldi:` Make sure kaldi is installed and compiled!

`2. cmake`

**2. Compile**

2.1 Initialization

`init_project.sh`

To init the workspace of cvte decoder, Its main job is copy link libraries and head files from kaldi.
```powershell
#Usage:
	./init_project.sh kaldi-root-path final.mdl-path HCLG.fst-path words.txt-path
```

2.2 Compile and install
```powershell
	mkdir build && cd build
	cmake ..
	make && make install
```
The main CmakeLists.txt is `src/CMakeLists.txt`, There is some directory path need to modify. Before run `cmake ..`, you should modify  `PROJECT_PATH` and  `KALDI_ROOT_PATH` to you own directory path. Targets(***cvte-decoder and oneline-cvte-decoder***) is installed to `PROJECT_PATH/bin`
	
**3. Usage**
```powershell
#Usage:
    # sli_split and word_split are install in PROJECT_PATH/spkdecoders/bin
	./sli_split final.mdl HCLG.fst 2017_03_07_16.57.40_2562.wav words.txt 15 output-path
	./word_split final.mdl HCLG.fst 2017_03_07_16.57.40_2562.wav words.txt 8 output-path 
For batch operation, you can run follow command:
    #That shell is alse in dir PROJECT_PATH/bin 
    ./split.sh data-path
```
