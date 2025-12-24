# skeletal-pose-to-activity-classifier
Code repository for Effects of Variable Pose Input on Neural Network Model Performance for Classifying Basketball Player Activity research paper.


This repository contains code for stratified KFold training of both the GCN and encoder-decoder transformer models outlined in the paper. Note: the same model architectures were applied to each of the joint combinations tested along with hyperparameter tuning in order to determine the best joint combinations to apply stratified KFold analysis.

## Data Format
The data used in this analysis is confidential and owned by the NBA. It contains skeletal tracking data where each row is a separate basketbal event. This work analyzes classifying dribbles, passes, shots, and rebounds over a 21 frame window using a wide variety of different joint combinations to analyze the impact of reducing joints on classification model performance. For this paper, 155 different joint combinations were tested and we analyzed over 400,000 event occurences.

The input dataframe are parquet files with the following columns of interest:

['game_id', 'stadium_id', 'player_id', 'team_id', 'event_seq_id', 'event_type', 'ball_position', 'player_com', 'pose_data', 'label']


## Model Architecture
We used a encoder-decoder transformer as outlined in transformer_kfold.ipynb file and a Graph Convolutional Network with GRU temporal encodings as outlined in GCN_kfold.ipynb file. For the GCN we define nodes as the skeletal joint coordinates and edges as the limb connections between joints.

## Hyperparameter Tuning
We use the software package Optuna for each joint pair combination to determine optimal hyperparameters to train each model. 
- **Transformer search space**
  - `d_model ∈ {64, 128, 256, 512}`
  - `n_head ∈ {4, 8, 16}`
  - `num_layers ∈ {2, 4, 6}`
  - `learning rate` searched on a **log-scaled range** `[1e-5, 1e-3]`
  - `batch size = 64` (fixed)
  - `dropout = 0.1` (fixed)
- **GCN search space**
  - `hidden_dim ∈ {64, 128, 256}`
  - `dropout ∈ [0.1, 0.3]` 
  - `learning rate` searched on a **log-scaled range** `[1e-5, 1e-3]`
  - `batch size ∈ {16, 32, 64}`

## Model Parameters
For the encoder-decoder transformer model and for the GCN model, the total number of learnable parameters varied depending on which hyperparamters were selected for each joint pair combination.
- **Transformer (encoder–decoder)**
  - Learnable Parameter Range: **0.86 M → 23.16 M**
  - Smallest: **singular joint (6 features)**; `d_model=64`, `num_layers=2`, `n_head=4` → ~**0.86 M**
  - Largest: **16 joints (51 features)**; `d_model=512`, `num_layers=6`, `n_head=16` → ~**23.16 M**

- **GCN**
  - Learnable Parameter Range: **0.04 M → 0.63 M**
  - Smallest: `hidden_dim=64` → ~**0.04 M**
  - Largest: `hidden_dim=256` → ~**0.63 M**

