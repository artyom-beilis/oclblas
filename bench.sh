#for M in 0 2 4 8 16 24 32 48 64 96 128 192 256 384 512 1024 2048
EXTRA="-i 10000 -w 10000"
randv()
{
    python -c 'import math; import random ; print int(math.ceil(math.pow(2,random.random()*12)))'
}
#for M in 0 1 2 4 8 16 32 64 128 256 512 1024 2048
for step in {0..50}
do
    if [ "$step" == "0" ]
    then
        K=0
        M=0
        N=0
    else
        M=$(randv)
        N=$(randv)
        K=$(randv)
        let prod=( $M '*' $N '*' $K )
        if [ $prod -lt 40960 ]
        then
            continue
        fi
    fi
    if [ "$K" == 0 ]
    then
        printf '%20s,'  'M:N:K'
    else
        printf '%20s,' "$M:$N:$K"
    fi
    for backend in cublas clblast clblas viennacl my mycuda
    do

        export TILE_SIZE=32
        export TILE_SIZE_K=8
        export BLOCK_X=4
        export BLOCK_Y=4

        if [ "$K" == 0 ]
        then
            printf '%18s, ' $backend
        else
            EXTRA=""
            if [ $backend == cpu ] && [ $M -gt 512 ] 
            then
                #printf '%10s, ' 'NA'
                #continue
                EXTRA="$EXTRA -i 10"
            fi
            v=$(./test -v $backend -m $M -n $N -k $K $EXTRA 2>/dev/null | awk '{print $1}')
            if [ "$backend" == "cublas" ]
            then
                ref="$v"
            fi
            rat=$(python -c "print ($v * 100.0 / $ref)" )
            printf '%10.2f/%-5.2f%%, ' $v $rat
        fi
    done
    printf '\n'
done    

