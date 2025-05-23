#!/bin/bash
trap : SIGTERM SIGINT

function abspath() {
    # generate absolute path from relative path
    # $1     : relative filename
    # return : absolute path
    if [ -d "$1" ]; then
        # dir
        (cd "$1"; pwd)
    elif [ -f "$1" ]; then
        # file
        if [[ $1 = /* ]]; then
            echo "$1"
        elif [[ $1 == */* ]]; then
            echo "$(cd "${1%/*}"; pwd)/${1##*/}"
        else
            echo "$(pwd)/$1"
        fi
    fi
}

roscore &
ROSCORE_PID=$!
sleep 1

rviz -d ../config/rviz/fusionforce_rough.rviz &
RVIZ_PID=$!

FUSIONFORCE_DIR=$(abspath "../../")

docker run \
  -it \
  --rm \
  --gpus all \
  --net=host \
  -v ${FUSIONFORCE_DIR}:/root/catkin_ws/src/fusionforce/ \
  -v $HOME/.cache/torch/:/root/.cache/torch/ \
  agishrus/fusionforce \
  /bin/bash -c \
  "cd /root/catkin_ws/; \
  catkin config \
      --cmake-args \
      -DCMAKE_BUILD_TYPE=Release; \
      catkin build; \
      source devel/setup.bash; \
      roslaunch fusionforce_demos fusionforce_rough.launch rviz:=false robot:=marv"

wait $ROSCORE_PID
wait $RVIZ_PID

if [[ $? -gt 128 ]]
then
    kill $ROSCORE_PID
    kill $RVIZ_PID
fi
