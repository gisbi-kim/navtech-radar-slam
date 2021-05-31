#!/bin/bash

root='/workspace/raid/krb/oxford-radar-robotcar-dataset/'

dirs=$(ls $root)

declare -a dir_array

i=0
for d in $dirs
do
	if [ -d "$root$d" ]; then
		dir_array[$i]=$d;
		let i++;
	fi
done

j=0
while [ $j -le $i ]
do
	echo ${dir_array[$j]};
	./build/yeti/odometry ${dir_array[$j]} ${dir_array[$j]} &
	./build/yeti/odometry ${dir_array[$(( $j + 1 ))]} ${dir_array[$(( $j + 1 ))]} &
	./build/yeti/odometry ${dir_array[$(( $j + 2 ))]} ${dir_array[$(( $j + 2 ))]} &
	./build/yeti/odometry ${dir_array[$(( $j + 3 ))]} ${dir_array[$(( $j + 3 ))]} &
	./build/yeti/odometry ${dir_array[$(( $j + 4 ))]} ${dir_array[$(( $j + 4 ))]} &
	j=$(( $j + 5 ));
	wait;
done
