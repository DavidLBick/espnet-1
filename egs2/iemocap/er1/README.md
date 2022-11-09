## Running 5 fold Evaluation with IEMOCAP

# Step 1: Set the IEMOCAP path in db.sh. Then run data prep 
```bash
local/data.sh --stage 1 --stop_stage 5
```
# Step 2: Run feature extraction for all 5 folds
```bash
local/prep_feats.sh <fold-number>
local/prep_feats.sh 1
```

# Step 3: Then create the tokenized list of discrete emotions 
 
```bash
./run_disc.sh --stage 5 --stop_stage 5
```

# Step 4: Extract Stats, Training and Inference
# Run stats extraction and training for each fold
# If you have 5 V100-32 GPUs, you can use bash parallel or xargs to train the models in parallel

# For continuous feature prediction, use 
```bash
./run_cts.sh <fold-number>
./run_cts.sh 1
```

