# Rockfish

Rockfish is the deep learning based tool for detecting 5mC DNA base modifications.

## Requirements

* ONT Guppy (basecalling) 
* Python >=3.9
* CUDA (for GPU inference)

## Installation

0. Create new environment (e.g. Conda):
   ```shell
   conda create --name rockfish
   ```

1. Clone the repository
   ```shell
   git clone ... rockfish && cd rockfish
   ```

2. Run installation
   ```shell
   pip install --extra-index-url https://download.pytorch.org/whl/cu113 .
   ```
   Note: "cu113" installs PyTorch for CUDA 11.3. If you want to install PyTorch for other CUDA version, replace "cu113" with appropriate version (e.g. for CUDA 10.2 "cu102").

3. Download models
   Available models: ***base***, ***small*** (or both with ***all***)
   ```shell
   rockfish download -m {all, base, small}
   ```


## Inference

1. Guppy basecalling
   ```shell
   guppy_basecaller -i <fast5_folder> -r -s <save_folder> --config <config_file> --fast5_out --device <cpu_or_gpus>
   ```
   Note: ```--fast5_out``` will output fast5 files with fastq and move_table groups needed for inference. This parameter is mandatory.

2. Run inference
   ```shell
   rockfish inference -i <saved_fast5_files> --reference <reference_path> --model_path <model_path> -r -t <n_workers> -b <batch_size> -d <devices>
   ```
   * Number of workers ```-t``` sets number of processes for generating the data.
   * Batch size ```-b``` is an optional parameter with default value of $4096$. However, for some GPUS (like V100 or A100 with 32GB/42 GB VRAM), it's appropriate to set it to a higher value (e.g. $n_{gpu} \times 8192$ or $n_{gpu} \times 16384$ for base model).
   * Examples of device parameter ```-d```:
     * CPU: No parameter
     * 1 GPU: ```-d 0```
     * 2 GPUs: ```-d 0,1```
     * 2 GPUs (3rd and 4th GPU): ```-d 2,3```

## Models
| Model | Encoder layers | Decoder layers | Features | Feedforward | Download link |
|-------|----------------|----------------|----------|-------------|---------------|
| Base  | 12             | 12             | 256      | 1024        |               |
| Small | 6              | 6              | 128      | 1024        |               |

## Example
1. Download data available at Nanopolish [website](https://nanopolish.readthedocs.io/en/latest/quickstart_call_methylation.html):
```shell
wget http://s3.climb.ac.uk/nanopolish_tutorial/methylation_example.tar.gz
tar -xvf methylation_example.tar.gz
cd methylation_example
```

2. Run guppy basecaller:
```shell
guppy_basecaller -i fast5_files/ -s basecalled -r --fast5_out -c dna_r9.4.1_450bps_sup.cfg -x "cuda:0"
```

3. Download T2T CHM13 Assembly (v2.0):
```shell
wget https://s3-us-west-2.amazonaws.com/human-pangenomics/T2T/CHM13/assemblies/analysis_set/chm13v2.0.fa.gz
```

4. Run inference
```shell
rockfish inference -i basecalled/workspace/ --model_path /raid-ssd/stanojevicd/dna-mod/final_training/small/small_final.ckpt -d 0 -t 32 --reference chm13v2.0.fa.gz
```
Result of the inference is ***predictions.tsv*** file. It is tab-delimited text file with four fileds:
  1. Read-id
  2. Contig name
  3. Position in contig
  4. Logit 

Evaluation on sample data:
1. Only intersection between ONT tools are used for evaluation
2. Only positions that have >= 5x are used for evaluation (NA12878 WGBS)
3. Correlation between ONT tools and WGBS: >= 5x

Read-level (n=258993) and correlation (n=64293) evaluation:

| Tool              | Accuracy | Precision | Recall   | FPR      | F1-score | Pearson's r |
|-------------------|----------|-----------|----------|----------|----------|-------------|
| Nanopolish        | 0.851861 |  0.757576 | 0.839686 | 0.141718 | 0.796521 | 0.819401    |
| Megalodon         | 0.885043 |  0.804421 | 0.881372 | 0.113021 |   0.8411 | 0.870277    |
| Rockfish  (small) | 0.943265 |  0.900087 | 0.940043 | 0.055036 | 0.919631 | 0.893       |

## Acknowledgement

This work has been supported in part by AI Singapore under the project the Deep Generative Modeling of Epigenomics Data (AISG-RPKS-2019-001), by Croatian Science Foundation under the project Single genome and metagenome assembly (IP-2018-01-5886), by National Supercomputing Centre, Singapore under the project Deep Generative Modelling of Epigenomics Data (13002174), by Epigenomics and Epitranscriptomics Research seed grant by Genome Institute of Singapore and by the A*STAR Computational Resource Centre through the use of its high-performance computing facilities.