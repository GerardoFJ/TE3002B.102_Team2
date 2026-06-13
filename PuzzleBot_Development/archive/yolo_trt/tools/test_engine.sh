#!/bin/bash
# Measure yolo_trt_node fps for a given engine and grab a sample frame.
# Usage: test_engine.sh <engine_path> <tag>
source /opt/ros/humble/setup.bash
source ~/cyclone_ws/install/setup.bash
source ~/PuzzlebotMain/PuzzleBot_Development/install/setup.bash
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

ENG="$1"; TAG="$2"
pkill -f yolo_trt_node 2>/dev/null; sleep 2
ros2 run yolo_trt yolo_trt_node --ros-args -p engine_path:="$ENG" \
    > "/tmp/yolo_$TAG.log" 2>&1 &
NPID=$!
sleep 30
python3 ~/PuzzlebotMain/PuzzleBot_Development/yolo_trt/tools/grab_frame.py \
    "/tmp/yolo_$TAG.jpg" > "/tmp/grab_$TAG.log" 2>&1
kill "$NPID" 2>/dev/null
sleep 1
echo "===== RESULT $TAG ($ENG) ====="
grep -E "fps|fatal|error" "/tmp/yolo_$TAG.log" | tail -4
cat "/tmp/grab_$TAG.log"
