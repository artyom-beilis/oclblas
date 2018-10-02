#for M in 0 2 4 8 16 24 32 48 64 96 128 192 256 384 512 1024 2048
EXTRA="-i 10000 -w 10000"
#for M in 0 1 2 4 8 16 32 64 128 256 512 1024 2048
for M in 0 32 64 128 256 512 1024 2048 4096
do
    if [ "$M" == 0 ]
    then
        printf '%5s,'  'M'
    else
        printf '%5d,' $M
    fi
    for backend in cpu cublas clblast clblas viennacl my mycuda
    do

        if (( $M < 256)) ; then
            export TILE_SIZE=16
            export TILE_SIZE_K=8
            export BLOCK_X=2
            export BLOCK_Y=2
        elif (( $M == 256)) ; then
            export TILE_SIZE=32
            export TILE_SIZE_K=16
            export BLOCK_X=4
            export BLOCK_Y=4
        else
            export TILE_SIZE=64
            export TILE_SIZE_K=16
            export BLOCK_X=8
            export BLOCK_Y=4
        fi

	# intel
        #    export TILE_SIZE=64
        #    export TILE_SIZE_K=8
        #    export BLOCK_X=8
        #    export BLOCK_Y=4

        if [ "$M" == 0 ]
        then
            printf '%10s, ' $backend
        else
            EXTRA=""
            if [ $backend == cpu ] && [ $M -gt 512 ] 
            then
                #printf '%10s, ' 'NA'
                #continue
                EXTRA="$EXTRA -i 10"
            fi
            v=$(./test -v $backend -m $M $EXTRA 2>/dev/null | awk '{print $1}')
            printf '%10.2f, ' $v 
        fi
    done
    printf '\n'
done
    

