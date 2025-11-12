# skeletal-pose-to-activity-classifier
Code repository for Effects of Variable Pose Input on Neural Network Model Performance for Classifying Basketball Player Activity paper.


This repository contains code for stratified KFold training of both the GCN and Transformer models outlined in the paper. Note the same model architectures along with hyperparameter tuning were applied to each of the joint combinations tested in order to determine the best joint combinations to apply stratiffied KFold analysis on.

## Data Format
The data used in this analysis is confidential and owned by the NBA. It contains skeletal tracking data where each row is a separate basketbal event. This work analyzes classifying dribbles, passes, shots, and rebounds over a 21 frame window using a wide variety of different joint combinations to analyze the impact of reducing joints on classification model performance. 155 different joint combinations were tested and we collected over 400,000 different event occurences.

The input dataframe are parquet files with the following columns of interest:

['game_id', 'stadium_id', 'player_id', 'team_id', 'event_seq_id', 'event_type', 'ball_position', 'player_com', 'pose_data', 'label']


## Transformer Architecture

## GCN Architrecture
