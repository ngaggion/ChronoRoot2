export nnUNet_raw="nnUNet_raw"
export nnUNet_preprocessed="nnUNet_preprocessed"
export nnUNet_results="nnUNet_results"

nnUNetv2_plan_and_preprocess -d 789 --verify_dataset_integrity

nnUNetv2_train 789 2d 0 
nnUNetv2_train 789 2d 1 
nnUNetv2_train 789 2d 2 
nnUNetv2_train 789 2d 3 
nnUNetv2_train 789 2d 4 

