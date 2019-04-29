#!/bin/bash 
M=$(echo 16 '*' $TILE_SIZE_M | bc)
K=$(echo 4096 / $TILE_SIZE_K '*'  $TILE_SIZE_K | bc)
timeout 5s ./test -v my -c -m $M -n $M -k $K -w 10 -a 1 -i 50 -P 2 2>/tmp/param.txt 1>/tmp/res.txt
echo $(cat /tmp/res.txt) $(cat /tmp/param.txt | head -n 1)
