## Running 5 fold Evaluation with IEMOCAP

### Step 1: Set the IEMOCAP path in db.sh. Then run data prep 
```bash
local/data.sh --stage 1 --stop_stage 5
```
### Step 2: Run feature extraction,tokenization for all 5 folds

```bash
local/prep_feats.sh <fold-number>
```
For example, to do this for fold 1, we can run 
```bash
local/prep_feats.sh 1
```

### Step 3: Extract Stats, Training and Inference
#### If you have 5 V100-32 GPUs, you can use bash parallel or xargs to train the models in parallel

#### For DISCRETE EMOTION:

#### The important file for this stage is run_disc.sh. This file runs the model training for fold <k>, given the fold number. So you will have to run the file as many times as there are number of folds. If you look into this file, there are a number of defined parameters. For example, 'er_config' which is where the model encoder and decoder architecture are defined. You can keep that as is, since the file defined in there is the one used to train the fold for our ICASSP submission. To run stage 6 until stage 9, run the run_disc.sh file as it is. After successfully running until stage 9, check for the results in the following file: exp/er_<er-tag>/inference_er_model_valid.acc.ave_10best/<test-dir>/report.txt.
```bash
./run_disc.sh <fold-number>
```

#### For CONTINUOUS EMOTION:
#### Similarly for continuous prediction, the important file to look into is run_cts.sh file. 
```bash
./run_cts.sh <fold-number>
./run_cts.sh 1
```

#### MTL training, inference. 
#### stage 6 only needs to run once for all folds and for both the discrete and contiuous emotion files. So to train MTL model, run the run_mtl.sh file from stage 7 to stage 9. This is already set in the file. 
```bash
./run_cts.sh <fold-number>
```

#### HMTL Training, inference.
#### To use the continuous predictions to predict discrete, run the following file: 
```bash
./run_hmtlcd.sh <fold-number>
```
#### To use the discrete prediction for continuous prediction, run the following:
```bash
./run_hmtldc.sh <fold-number>
```

---------------------------------------------------------------------------------
#### Its important to explain in detail one of the config files we have used. So for this I will go over the conf/train_hubert_ll60k_conformer_discrete.yaml (used in run_dics.sh). The encoder artitecture is defined as conformer, The decoder architecture is defined as mtl_decoder. which can be found in '../../../espnet2/er/decoder/seq_classification.py'. The pool_type is attention pooling which is also defined in the same file.The decoder architecture is what is varied over the MTL, HMTLCD and HTMLDC architectures. If you need to study this further, look into the config files used for each architectures and look into the decoder architecture. 
