## Training Emotion Recognition Models with MSPPODCAST

### Step 1: Set the MSPPODCAST path in db.sh. 

### Step 2: Data Prep,Feature Extraction, Tokenization and Stats Creation
 
```bash
local/run_mtl.sh --stage 1 --stop_stage 6
```

### Step 3: Model Training and Inference

#### DISCRETE model: 
#### Look into the run_disc.sh file. For further explanation for the important parameters, look into the README for iemocap recipe. 
```bash
./run_disc.sh
```

#### CONTINUOUS model: 
```bash
./run_cts.sh
```

#### MTL model:
```bash
./run_mtl.sh
```

### HMTL DC model:
```bash
./run_hmtldc.sh
```

### HMTL CD model:
```bash
./run_hmtlcd.sh
```

## Applying to a New Corpora 

The first step is to prepare the data. Please look into the [Kaldi toolkit](https://kaldi-asr.org/doc/data_prep.html).
Create data to have the following format: 

```bash
data 
  ->train
      -> wav.scp 
      -> segments
      -> feats.scp
      -> utt2spk
      -> spk2utt
      -> emotion_cts 
  -> valid 
      -> wav.scp 
      -> segments
      -> feats.scp
      -> utt2spk
      -> spk2utt
      -> emotion_cts 
```

The script to do so is placed in local/data.sh of the recipe.
The emotion_cts file needs to contain lines with 

```bash
<utt-id> <valence in float> <arousal in float> <dominance in float>
```

- Copy and modify the run script run\*.sh scripts for your corpus. 
- For a new dataset, you would need to write your own version of local/data.sh which creates the data in the Kaldi format including the emotion file. 
- Then you modify the train_set,valid_set, and inference_sets options in run\*.sh files depending on the names of your subsets. 


