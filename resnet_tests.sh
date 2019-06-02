export GLOG_minloglevel=3
BATCH=16
for tester in test_conv_cudnn test_conv_caffe_cuda  test_conv_caffe_ocl
do
    TEST=./$tester
    DEV=0

    echo "============ $tester =============== "

    echo "ResNet 50"

    $TEST  -p $DEV -c  -B $BATCH -C 128 -D 28  -P 0 -K 1 -T 2 -N 256
    $TEST  -p $DEV -c  -B $BATCH -C 128 -D 28  -P 1 -K 3 -T 1 -N 128
    $TEST  -p $DEV -c  -B $BATCH -C 128 -D 28  -P 1 -K 3 -T 2 -N 256
    $TEST  -p $DEV -c  -B $BATCH -C 256 -D 14  -P 0 -K 1 -T 2 -N 512

    $TEST  -p $DEV -c  -B $BATCH -C 256 -D 14  -P 1 -K 3 -T 1 -N 256
    $TEST  -p $DEV -c  -B $BATCH -C 256 -D 14  -P 1 -K 3 -T 2 -N 512
    $TEST  -p $DEV -c  -B $BATCH -C 3   -D 224 -P 3 -K 7 -T 2 -N 64
    $TEST  -p $DEV -c  -B $BATCH -C 512 -D 7   -P 1 -K 3 -T 1 -N 512

    $TEST  -p $DEV -c  -B $BATCH -C 64  -D 56  -P 0 -K 1 -T 1 -N 64
    $TEST  -p $DEV -c  -B $BATCH -C 64  -D 56  -P 0 -K 1 -T 2 -N 128
    $TEST  -p $DEV -c  -B $BATCH -C 64  -D 56  -P 1 -K 3 -T 1 -N 64
    $TEST  -p $DEV -c  -B $BATCH -C 64  -D 56  -P 1 -K 3 -T 2 -N 128

    echo "AlexNet"

    $TEST  -p $DEV -c  -B $BATCH -C 3   -D 224  -P 0 -K 11 -T 4 -N 96
    $TEST  -p $DEV -c  -B $BATCH -C 48  -D 55   -P 2 -K 5  -T 2 -N 256  # due to grouping
    $TEST  -p $DEV -c  -B $BATCH -C 256 -D 27   -P 1 -K 3  -T 1 -N 384
    $TEST  -p $DEV -c  -B $BATCH -C 192 -D 13   -P 1 -K 3  -T 1 -N 384
    $TEST  -p $DEV -c  -B $BATCH -C 192 -D 13   -P 1 -K 3  -T 1 -N 256

done

