# Download data from Nanopolish
wget http://s3.climb.ac.uk/nanopolish_tutorial/methylation_example.tar.gz
tar -xzf methylation_example.tar.gz
cd methylation_example

# Run Guppy Basecaller
$GUPPY_PATH -i fast5_files/ -s basecalled -r --fast5_out -c dna_r9.4.1_450bps_sup.cfg -x "cuda:0"

# Download T2T CHM13 Assembly (v2.0)
wget https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/analysis_set/chm13v2.0.fa.gz

# Download small model
rockfish download -m small -s .

# Run inference
rockfish inference -i basecalled/workspace/ --model_path rf_small.ckpt -d 0 -t 8 --reference chm13v2.0.fa.gz