#!/bin/bash
filename=$1
echo "filename="$filename;

while read myline
do
    arr=$myline;
#subarr=${arr#*|}   # Get the first char after |
#    rarr=${subarr%|*}  # Get the char before last |
#    echo $subarr
   echo ${arr%%|*}
done < $filename
