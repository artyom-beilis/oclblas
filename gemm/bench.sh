#for M in 0 2 4 8 16 24 32 48 64 96 128 192 256 384 512 1024 2048

# nvidia
#DEV=1
#BCK="cublas cutlass clblas clblast my"

# radeon
#DEV=2
#BCK="miopengemm clblas clblast my"

# intel
#DEV=0
#BCK="cpu clblas clblast viennacl my"

# nvidia only
DEV=0
BCK="cublas clblas clblast my mycuda"

EXTRA="-i 100 -w 100 "
randv()
{
    python -c 'import math; import random ; print int(math.ceil(math.pow(2,random.random()*12)))'
}

pow2()
{
	python -c "import math; import random ; print int(math.pow(2,$1))"
}

#for step in {0..12}
for M in 0 16 32 64 128 256 512 1024 2048
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
            v=$(./test_gemm_$backend -m $M -n $N -k $K $EXTRA -P $DEV 2>/dev/null | awk '{print $1}')
            printf '%10.2f, ' $v
        fi
    done
    printf '\n'
done    

