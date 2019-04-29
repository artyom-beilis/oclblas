B=64
for PARAMS in   "-C 3 -D 227 -T 4 -K 11 -N 96" \
		"-C 96 -D 55 -P 2 -K 5 -N 256"  \
		"-C 256 -D 13 -P 1 -K 3 -N 384" 
do
	echo $PARAMS
	for PROG in ./test_conv ./test_conv_hip
	do
		echo $PROG
		$PROG -c -p 1 -B $B $PARAMS 2>/dev/null
	done
done

