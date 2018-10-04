rm -f log_full.txt
EXTRA="-c -i 100 -w 100"
for M in "-m 8192 -n 1024 -k 32"
do
    rm -f log.txt
    for TSK in 1 2 4 8 16 32 64 128
    do
        for TS in 8 16 32 64 
        do
            for BSX in 4 8 16
            do
                for BSY in 4 8 16 
                do
                    if (( $BSX > $TS ))
                    then
                        continue
                    fi
                    if (( $BSY > $TS ))
                    then
                        continue
                    fi
                    if (( $TS * $TSK / $BSX / $BSY < 16 ))
                    then
                        continue
                    fi
                    if (( $TS * $TSK * 8 > 32768 ))
                    then
                        continue
                    fi

                    if (( $TS * $TS / $BSX / $BSY < 32 ))
                    then
                        continue
                    fi

                    #if (( $M >= 512 ))
                    #then
                    #    EXTRA="-i 500 -w 500"
                    #fi
                        
                    export TILE_SIZE=$TS
                    export TILE_SIZE_K=$TSK
                    export BLOCK_X=$BSX
                    export BLOCK_Y=$BSY
                    VAL=$(timeout 5 ./test $M -v my  $EXTRA 2>/dev/null)
                    printf "%20s TS=$TS; TSK=$TSK; BX=$BSX; BY=$BSY\n" "$VAL" | tee -a log.txt 
                done
            done
        done
    done
    
    echo ----------------
    echo Results $M: | tee -a log_full.txt
    sort -n log.txt | tail -n 5 | tee -a log_full.txt
    echo ----------------
done
