#!/bin/bash

docker build docker \
             -t bbq_image \
             --build-arg UID=${UID} \
             --build-arg GID=${UID}
