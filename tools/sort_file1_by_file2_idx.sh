#! /bin/bash

awk 'FNR==NR{a[$1]=$0;next} $1 in a {print a[$1]}' $1 $2

----------------------------------------------------------------------------------
tools//sort_file.sh
#! /bin/bash

export LC_ALL=C

file=$1
sort -k1,1 -u <$file >$file.tmp
mv $file.tmp $file


