#!/bin/bash
if [ "$#" -ne 6 ]; then
    echo "Usage: <# of queens> <gamma> <start_threshold> <end_threshold> <# of samples per threshold> <output_file>"
    echo " "
    echo "In general pick <start_threshold> to be at least <# of queens> * <threshold>"
    echo " "
    exit
fi
NUM_QUEENS=$1
GAMMA=$2
START_THRESHOLD=$3
END_THRESHOLD=$4
REPEAT_N_TIMES=$5
OUTPUT_FILE=$6

echo ""
echo "Running nQueens for"
echo "    n             = $NUM_QUEENS queens"
echo "    gamma         = $GAMMA"
echo "    threshold     = {$START_THRESHOLD..$END_THRESHOLD}"
echo "    samples per   = $REPEAT_N_TIMES"
echo "    threshold"
echo ""
echo "Storing results in $OUTPUT_FILE"
echo ""
make && for threshold in `seq $START_THRESHOLD $END_THRESHOLD`; do
    for repeat in {1..$REPEAT_N_TIMES}; do
        bin/release/queens_driver $NUM_QUEENS $GAMMA $threshold;
    done;
done > $OUTPUT_FILE
echo ""
