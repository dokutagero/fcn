#!/bin/bash

docker run -it --runtime=nvidia --rm -v /home/ubuntu/workspace/fcn_root:/root/fcn -v /home/ubuntu/data:/root/data --name=mychainer mychainer /bin/bash
