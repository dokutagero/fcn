#!/bin/bash

docker run -it --runtime=nvidia --rm -v /home/ubuntu/workspace/fcn:/root/fcn -v /home/ubuntu/workspace/chainer:/root/chainer -v /home/ubuntu/workspace/data:/root/data --name=mychainer mychainer /bin/bash
