#!/bin/bash

# Download data from Nanopolish
wget https://www.dropbox.com/s/4ga4j1diftabn8n/example.tar.gz?dl=0 -O example.tar.gz
tar -xzf example.tar.gz
cd example

# Download small model
rockfish download -m small -s .

# Run inference
if [[ -z ${GPUS} ]]; then
  rockfish inference -i fast5/ --model_path rf_small.ckpt -t 2 --reference chm13v2.0_chr20.fa.gz
else
  rockfish inference -i fast5/ --model_path rf_small.ckpt -d ${GPUS} -t 2 --reference chm13v2.0_chr20.fa.gz
fi
