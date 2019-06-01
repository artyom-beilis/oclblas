TEST=./test_conv_hip
METHOD=miopen
DEV=0


$TEST -v $METHOD -p $DEV -c  -B 16 -C 128 -D 28  -P 0 -K 1 -T 2 -N 256
$TEST -v $METHOD -p $DEV -c  -B 16 -C 128 -D 28  -P 1 -K 3 -T 1 -N 128
$TEST -v $METHOD -p $DEV -c  -B 16 -C 128 -D 28  -P 1 -K 3 -T 2 -N 256
$TEST -v $METHOD -p $DEV -c  -B 16 -C 256 -D 14  -P 0 -K 1 -T 2 -N 512

$TEST -v $METHOD -p $DEV -c  -B 16 -C 256 -D 14  -P 1 -K 3 -T 1 -N 256
$TEST -v $METHOD -p $DEV -c  -B 16 -C 256 -D 14  -P 1 -K 3 -T 2 -N 512
$TEST -v $METHOD -p $DEV -c  -B 16 -C 3   -D 224 -P 3 -K 7 -T 2 -N 64
$TEST -v $METHOD -p $DEV -c  -B 16 -C 512 -D 7   -P 1 -K 3 -T 1 -N 512

$TEST -v $METHOD -p $DEV -c  -B 16 -C 64  -D 56  -P 0 -K 1 -T 1 -N 64
$TEST -v $METHOD -p $DEV -c  -B 16 -C 64  -D 56  -P 0 -K 1 -T 2 -N 128
$TEST -v $METHOD -p $DEV -c  -B 16 -C 64  -D 56  -P 1 -K 3 -T 1 -N 64
$TEST -v $METHOD -p $DEV -c  -B 16 -C 64  -D 56  -P 1 -K 3 -T 2 -N 128



