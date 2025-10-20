#!/bin/bash
# Additional Docker commands
xhost +local:
docker start "safae_botsort_container"
docker exec -it -e DISPLAY=$DISPLAY -w "/app/main" "safae_botsort_container" /bin/bash