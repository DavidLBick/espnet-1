## Training Emotion Recognition Models with MSPPODCAST

# Step 1: Set the MSPPODCAST path in db.sh. 

# Step 2: Data Prep,Feature Extraction, Tokenization and Stats Creation
 
```bash
local/run_mtl.sh --stage 1 --stop_stage 6
```

# Step 3: Model Training and Inference
# For continuous feature prediction, use 
```bash
./run_cts.sh <fold-number>
./run_cts.sh 1
```

