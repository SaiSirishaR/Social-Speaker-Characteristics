#!/bin/bash
pwd='path/to/featfolder/'
feat_folder="/pathto/arff"
new_feat_folder="/pathto/csv"

#mkdir $new_feat_folder
cd $feat_folder
for file in *;
do
          echo $file
          #mv "$file" ${file// /_}
          fbname=$(basename "$file" .arff)
          echo $fbname
          #ffmpeg -i $file $fbname.wav
          #ffmpeg -i $file $fbname.wav
          #mpg123 -w $fbname.wav $file
          #sox $file -r 16000 -c 1 $new_folder/$file
          cat $file | $pwd/arff2csv  > $new_feat_folder/$fbname.csv

done
