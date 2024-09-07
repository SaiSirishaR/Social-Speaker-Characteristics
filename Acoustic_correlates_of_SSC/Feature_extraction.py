#!/bin/bash

wav_folder="/path/to/wavefolder/"
feat_folder="/path/to/fetfolder/"
opensmile="/path/to/opensmiletoolkit/"

cd $wav_folder
for file in *;
do
echo $file

fbname=$(basename "$file" .wav)
/path/to/opensmiletoolkit/SMILExtract -C $opensmile/config/is09-13/IS12_speaker_trait.conf -I $wav_folder/$file -O $feat_folder/$fbname.arff

done
