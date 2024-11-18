# Rockfish

Rockfish is the deep learning based tool for detecting 5mC DNA base modifications.

## Requirements

* Linux (tested on Ubuntu 20.04)
* ONT Dorado (sup model; tested on v0.5.0 - [download](https://cdn.oxfordnanoportal.com/software/analysis/dorado-0.8.3-linux-x64.tar.gz))
* Python >= 3.9
* CUDA (for GPU inference; tested on 11.8)

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
   git clone -b r10.4.1 https://github.com/lbcb-sci/rockfish.git --single-branch rockfish && cd rockfish
   ```

2. Run installation
   ```shell
   pip install --extra-index-url https://download.pytorch.org/whl/cu118 .
   ```
   Note: "cu113" installs PyTorch for CUDA 11.8. If you want to install PyTorch for other CUDA version, replace "cu118" with appropriate version (e.g. for CUDA 10.2 "cu102"). For CPU version replace "cu118" with "cpu".

   #### Install Flash Attention (Optional)
   ```shell
   pip install flash-attn --no-build-isolation
   ```
   
   Installing [Flash Attention](https://github.com/Dao-AILab/flash-attention) can significantly speed up inference by optimizing attention mechanisms, reducing memory usage, and increasing efficiency without compromising accuracy.

   
   Installation should take a few minutes on a desktop computer with reasonable network bandwidth.

3. Download models
   Available models: ***5kHz***
   ```shell
   rockfish download -m {all, 5kHz} -s <save_path>
   ```


## Inference

1. Dorado    basecalling
   ```shell
   dorado basecaller -x <devices> -r --emit-moves <model> <pod5_files> > basecalls.bam
   ```
   Note: ```--emit-moves``` will output move table field for each entry in bam file. Move table is needed for inference. Any super-accurate model with the given data sampling frequency could be used for basecalling.

2. Run inference
   ```shell
   rockfish inference -i <pod5_files> --bam_path <bam_path> --model_path <model_path> -r -t <n_workers> -b <batch_size> -d <devices>
   ```
   * Number of workers ```-t``` sets number of processes for generating the data.
   * Batch size ```-b``` is an optional parameter with default value of $4096$. However, for some GPUS (like V100 or A100 with 32GB/42 GB VRAM), it's appropriate to set it to a higher value (e.g. $n_{gpu} \times 8192$ or $n_{gpu} \times 16384$ for base model).
   * Examples of device parameter ```-d```:
     * CPU: No parameter
     * 1 GPU: ```-d 0```
     * 2 GPUs: ```-d 0,1```
     * 2 GPUs (3rd and 4th GPU): ```-d 2,3```

## Models
| Model | Encoder layers | Decoder layers | Features | Feedforward | Dorado model              |
|-------|----------------|----------------|----------|-------------|---------------------------|
| 5kHz  | 12             | 12             | 256      | 2048        | $\geq$ dna_r10.4.1_e8.2_400bps_sup@v4.2.0 |

## Output
Result of the inference is ***predictions.tsv*** file. It is tab-delimited text file with four fileds:
  1. Read-id
  2. Position in the read
  3. 5mC probability (use `-l` flag to output logits instead of probabilities)


## Acknowledgement

This work has been supported in part by Croatian Science Foundation under the project Single genome and metagenome assembly (IP-2018-01-5886), by Epigenomics and Epitranscriptomics Research seed grant from Genome Institute of Singapore (GIS), by Career Development Fund (C210812037) from A*STAR, and by the A*STAR Computational Resource Centre through the use of its high-performance computing facilities.
