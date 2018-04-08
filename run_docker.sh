#!/bin/bash

docker run -it --runtime=nvidia --rm -v /home/ubuntu/workspace:/root -p 80:5000 --name=fcn fcn /bin/bash
