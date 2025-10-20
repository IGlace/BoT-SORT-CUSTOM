#!/usr/bin/env bash
HOST_DIR="${1:-$(pwd)}"

docker run -it -d \
  -v /tmp/.X11-unix/:/tmp/.X11-unix/ \
  -e DISPLAY="$DISPLAY" \
  -v "$HOST_DIR":/app/main \
  -v /dev/shm/:/dev/shm/ \
  --gpus all \
  --shm-size "120G" \
  --memory "100g" \
  --name "safae_botsort_container" \
  "botsort_python3.7_image"


# python3 tools/demo.py video --path assets/test16.mp4 -f yolox/exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_mot17.pth.tar --with-reid --fuse-score --fp16 --fuse --save_result
