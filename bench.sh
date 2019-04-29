#for M in 0 2 4 8 16 24 32 48 64 96 128 192 256 384 512 1024 2048

# nvidia
DEV=1
BCK="cublas cutlass clblas clblast"

# radeon
#DEV=2
#BCK="miopengemm clblas clblast"

# intel
#DEV=0
#BCK="cpu clblas clblast viennacl"

EXTRA="-i 10000 -w 10000 "
randv()
{
    python -c 'import math; import random ; print int(math.ceil(math.pow(2,random.random()*12)))'
}

pow2()
{
	python -c "import math; import random ; print int(math.pow(2,$1))"
}

#for step in {0..12}
for M in 0 16 32 64 128 256 512 1024 2048 4096
do
    if [ "$step" == "0" ]
    then
        K=0
        M=0
        N=0
    elif true
    then
    	N=$M
	K=$M
    else
	M=$(randv)
	N=$(randv)
	K=$(randv)
	M=$(pow2 $step)
	N=$(pow2 $step)
	K=$(pow2 $step)
	K=8192
    fi
    if [ "$K" == 0 ]
    then
        printf '%20s,'  'M:N:K'
    else
        printf '%20s,' "$M:$N:$K"
    fi
    for backend in $BCK
    do

        export TILE_SIZE_N=32
        export TILE_SIZE_M=64
        export TILE_SIZE_K=8
        export BLOCK_X=4
        export BLOCK_Y=4

        if [ "$K" == 0 ]
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
            v=$(./test -v $backend -m $M -n $N -k $K $EXTRA -P $DEV 2>/dev/null | awk '{print $1}')
            printf '%10.2f, ' $v
        fi
    done
    printf '\n'
done    

