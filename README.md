# Rockfish

## Update: R10.4.1 code can be found [here](https://github.com/lbcb-sci/rockfish/tree/r10.4.1) (branch r10.4.1).

Rockfish is the deep learning based tool for detecting 5mC DNA base modifications.

**Find the small example [here](#Example).**

## Requirements

* Linux (tested on Ubuntu 20.04)
* ONT Guppy >= 5 (sup model; tested on 5.0.14 - [download](https://cdn.oxfordnanoportal.com/software/analysis/ont-guppy-cpu_5.0.14_linux64.tar.gz))
* Python >= 3.9
* CUDA (for GPU inference; tested on 11.3)

### Python Requirements
Python requirements can be found in [setup.cfg](setup.cfg)

## Installation

0. a) Create new environment (e.g. Conda):
   ```shell
   conda create --name rockfish python=3.9
   ```
   
   b) Activate the environment
   ```shell
   conda activate rockfish
   ```

1. Clone the repository
   ```shell
   git clone https://github.com/lbcb-sci/rockfish.git rockfish && cd rockfish
   ```

2. Run installation
   ```shell
   pip install --extra-index-url https://download.pytorch.org/whl/cu113 .
   ```
   Note: "cu113" installs PyTorch for CUDA 11.3. If you want to install PyTorch for other CUDA version, replace "cu113" with appropriate version (e.g. for CUDA 10.2 "cu102"). For CPU version replace "cu113" with "cpu".

   Installation should take a few minutes on a desktop computer with reasonable network bandwidth.

3. Download models
   Available models: ***base***, ***small*** (or both with ***all***)
   ```shell
   rockfish download -m {all, base, small} -s <save_path>
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
| Model | Encoder layers | Decoder layers | Features | Feedforward | Guppy config              |
|-------|----------------|----------------|----------|-------------|---------------------------|
| Base  | 12             | 12             | 384      | 2048        | dna_r9.4.1_450bps_sup.cfg |
| Small | 6              | 6              | 128      | 1024        | dna_r9.4.1_450bps_sup.cfg |

## Example
Run the example script on 1000 pre-basecalled (Guppy 5.0.14. sup) fast5 files (sampled from Nanopolish [data](https://nanopolish.readthedocs.io/en/latest/quickstart_call_methylation.html)):
```shell
GPUS=<devices> ./scripts/example.sh
```
Note: Use GPUS var to set GPUs for the inference. Omit GPUS if the inference is run on CPU.

Result of the inference is ***predictions.tsv*** file. It is tab-delimited text file with four fileds:
  1. Read-id
  2. Contig name
  3. Position in the contig
  4. 5mC probability 

Note: The predictions can have a slightly different order compared to the predictions in the ***expected.tsv*** file due to multiprocessing. The running time should be about 5 minutes for CPU mode. GPU should be significantly faster.


## Acknowledgement

This work has been supported in part by Croatian Science Foundation under the project Single genome and metagenome assembly (IP-2018-01-5886), by Epigenomics and Epitranscriptomics Research seed grant from Genome Institute of Singapore (GIS), by Career Development Fund (C210812037) from A*STAR, and by the A*STAR Computational Resource Centre through the use of its high-performance computing facilities.
