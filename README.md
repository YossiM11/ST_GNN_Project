# Bridging Self-Supervised Clustering and Supervised Prediction in Spatial Transcriptomics

**Yale University CPSC 4830 - Final Project - Spring 2026**

**Author:** Yossi Moff (yossi.moff@yale.edu)

## Overview

This project extends GraphST (Long et al., Nature Communications 2023) with a novel semi-supervised training objective for spatial transcriptomics analysis. I introduce a joint loss function that combines GraphST's self-supervised contrastive pretraining with a supervised classification objective for cortical layer prediction.

### Adjustment to Loss Function

Original GraphST training loss, entirely self-supervised:

$$\mathcal{L} = \lambda_1 \mathcal{L}_{recon} + \lambda_2 (\mathcal{L}_{SCL} + \mathcal{L}_{SCL_{corrupt}})$$

where $\lambda_1=10$, $\lambda_2=1$ (fixed, based on values from paper), $L_{recon}$ is reconstruction loss, and $L_{SCL}$ is the spatial contrastive loss (including both real and corrupted graph).

**My semi-supervised extension:**

$$L = \lambda_1 \mathcal{L}_{recon} + \lambda_2 (\mathcal{L}_{SCL} + \mathcal{L}_{SCL_{corrupt}}) + \gamma \mathcal{L}_{cls}$$

where $L_{cls}$ is cross-entropy classification loss on labeled spots, and $\gamma \in \{0, 0.1, 0.5, 1.0, 5.0\}$ controls the supervision strength.
$\gamma=0$ reduces the loss function to the original GraphST loss function (purely unsupervised).
$\gamma>0$ introduces semi-supervised learning, with the classification loss exerting more influence on the overall loss function output as $\gamma$ increases.

## Results

Note: All results shown were run on DLPFC slice 151673, with an 80/20 train/test split. I ran a few baselines for classification, the most informative of which is applying an MLP head to frozen embeddings produced by the original GraphST architecture (row 5). The last four rows show the classification performance (Marco-F1 and AUROC) and clustering performance (ARI and NMI) of my novel Joint GraphST architecture. See report for discussion of results.

| Model | Macro-F1 | AUROC | ARI | NMI |
|-------|----------|-------|-----|-----|
| Raw counts → MLP | 0.5606 | 0.8957 | — | — |
| PCA → MLP | 0.6211 | 0.9270 | — | — |
| Supervised GCN | 0.1486 | 0.5041 | — | — |
| Supervised GIN | 0.1462 | 0.4908 | — | — |
| GraphST → MLP (frozen) | 0.8784 | 0.9905 | 0.4162 | 0.5322 |
| Joint GraphST γ=0.1 | 0.8174 | 0.9697 | 0.3900 | 0.5316 |
| Joint GraphST γ=0.5 | 0.8504 | 0.9776 | 0.4549 | 0.5641 |
| Joint GraphST γ=1.0 | 0.8727 | 0.9822 | 0.4558 | 0.5664 |
| **Joint GraphST γ=5.0** | **0.9100** | **0.9885** | **0.4542** | **0.5657** |

## How to Run the Code

Clone this repository:
git clone https://github.com/YossiM11/ST_GNN_Project.git
cd ST_GNN_Project

Clone GraphST into the project folder:
git clone https://github.com/JinmiaoChenLab/GraphST.git
cd GraphST && pip install . && cd ..

Set up the Python environment:
conda create -n GraphST python=3.10 -y
conda activate GraphST
pip install -r requirements.txt
python -m ipykernel install --user --name=GraphST --display-name "GraphST"

Download the DLPFC data:
- h5 files from: `https://spatial-dlpfc.s3.us-east-2.amazonaws.com/h5/{slice_id}_filtered_feature_bc_matrix.h5`
- Spatial positions from: `https://github.com/LieberInstitute/HumanPilot/tree/master/10X/{slice_id}`
- Annotations from STAGATE (Dong & Zhang, 2022) Google Drive: https://drive.google.com/drive/folders/1bkDfuq5YJmJOsdAEaFesp3CkxenXEemb
- Place in `GraphST/Data/{slice_id}/` with structure:
GraphST/Data/151673/
├── filtered_feature_bc_matrix.h5
├── metadata.tsv
└── spatial/
└── tissue_positions_list.csv

5. Open `project_notebook.ipynb` in Jupyter.

6. Select the **GraphST** kernel and press **Run All**.

## Repository Structure
ST_GNN_Project/
├── project_notebook.ipynb   # Main analysis notebook
├── README.md                # This file
├── requirements.txt         # Python dependencies
└── report.pdf               # Final report (see paper)

## References

1. Long et al. (2023). GraphST. Nature Communications.
2. Dong & Zhang (2022). STAGATE. Nature Communications.
3. Liu et al. (2024). Benchmarking GNNs for ST. CSBJ.
4. Kipf & Welling (2017). Semi-supervised GCN. ICLR.
5. You et al. (2020). Graph contrastive learning. NeurIPS.
6. Rong et al. (2020). Self-supervised graph transformer. NeurIPS.
