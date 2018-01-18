#!/bin/bash

docker run -it --runtime=nvidia --rm -v /home/ubuntu/workspace/fcn_root:/root/fcn --name=mychainer mychainer /bin/bash
