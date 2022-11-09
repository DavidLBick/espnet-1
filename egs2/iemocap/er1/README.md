## Running 5 fold Evaluation with IEMOCAP

# Step 1: Set the IEMOCAP path in db.sh. Then run data prep 
```bash
local/data.sh --stage 1 --stop_stage 5
```
# Step 2: Run feature extraction,tokenization for all 5 folds

```bash
local/prep_feats.sh <fold-number>
```
For example, to do this for fold 1, we can run 
```bash
local/prep_feats.sh 1
```

# Step 3: Extract Stats, Training and Inference
# If you have 5 V100-32 GPUs, you can use bash parallel or xargs to train the models in parallel

# For continuous feature prediction, use 
```bash
./run_cts.sh <fold-number>
./run_cts.sh 1
```
# For discrete prediction, 


