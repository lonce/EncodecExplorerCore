# Explore Encodec

This repository contains notebooks and resources for exploring [Encodec](https://arxiv.org/abs/2210.13438).

---

## Notebooks

- **[ExploreEncodecFiles.ipynb](ExploreEncodecFiles.ipynb)**  
  Just load audio files (from a Hugging Face dataset) and decode them with different numbers of codebooks. Visualize. Auralize. 

- **[ExploreEncodec.ipynb](ExploreEncodec.ipynb)**  
  Interactive exploration of API etc starting from a HuggingFace dataset of .ecdc files. 
  - (Even this only uses the basic Encodec API - there in no custom Pytorch dataset structure, DataLoader, etc)
  

---

## Datasets

The datasets used in this project are available here:  
[📂 Google Drive Folder](https://drive.google.com/drive/folders/1P1BZXeMle5Fpcdup6lYHZHROi0qo8OR2?usp=sharing) - the wav dataset is huge (you might just read in audio files yourself, no dataset needed). The ecdc dataset is not so big and .

---

## Environment

The repository includes an `environment.yml` file to set up the required Conda environment. It just has a bunch of packages I use for most of my generic audio work. 

### Create the environment

```bash
conda env create -f environment.yml -n basicaudio
```

### Activate it

```bash
conda activate basicaudio
```

---

## Getting Started

1. Clone this repository  
   ```bash
   git clone https://github.com/yourusername/explore-encodec.git
   cd explore-encodec
   ```

2. Download the datasets from the [Google Drive link](https://drive.google.com/drive/folders/1P1BZXeMle5Fpcdup6lYHZHROi0qo8OR2?usp=sharing).  

3. Launch Jupyter Lab and open one of the notebooks:  
   ```bash
   jupyter lab
   ```





lonce.org
