 #! /bin/bash

    function remove_blank() {
        for loop in `ls -1 | tr ' '  '#'`
        do  
            mv  "`echo $loop | sed 's/#/ /g' `"  "`echo $loop |sed 's/#//g' `"  2> /dev/null 
        done
	}

    function read_dir(){
		cd $1
		remove_blank
		cd -
        for file in `ls $1`       
        do
            if [ -d $1"/"$file ] 
            then
                read_dir $1"/"$file
            else
                echo $1"/"$file   
            fi
        done
    }   
    read_dir $1
-
