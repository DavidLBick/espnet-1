import sys
import opensmile
import numpy as np
from kaldiio import WriteHelper
import os

dir = sys.argv[1]

with open(os.path.join(dir, "wav.scp"), "r") as f:
    wav_scp = dict(
        map(lambda x: (x.strip().split(" ")[0], x.strip().split(" ")[1]), f.readlines())
    )


processor = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    sampling_rate=16000,
    resample=True,
    num_workers=36,
)

out_path = os.path.join(dir, "feats")
with WriteHelper("ark,scp:{}.ark,{}.scp".format(out_path, out_path)) as writer:
    for id, wav in wav_scp.items():
        feat = np.array(processor.process_file(wav))
        writer(id, feat)
