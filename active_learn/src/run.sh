#!/bin/bash

PROJ_PATH="/home/fxiong/thesis/active_learner"
echo "uploading file..."
scp *.py sg:${PROJ_PATH}
echo "start running..."
ssh -f -X sg "echo 'start' && cd ${PROJ_PATH} && nohup python -u img_class.py < /dev/null > run.log 2>&1 &"
echo "Job launched."
