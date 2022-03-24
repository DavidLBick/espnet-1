import csv
import argparse
import pdb
import pickle
import os 
import numpy as np

UTT_i = 1
SPK_i = 2
EMO_i = 3
SENT_i = 4
DIA_ID_i = 5
UTT_ID_i = 6
SEASON_i = 7
EPISODE_i = 8
START_TIME_i = 9
END_TIME_i = 10

def parse_csv(label_file, audio_dir, data_dir):
  with open("data/speaker_dict.pkl", 'rb') as f:
    spk_dict = pickle.load(f)
 
  wav_scp = []
  text = []
  utt2spk = []  
  with open(label_file) as csvfile:  
      csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
      next(csv_reader, None)  # skip header 
      for row in csv_reader:
        # adding speakerid as prefix of utterance id because of note in kaldi data-prep document: 
        # https://kaldi-asr.org/doc/data_prep.html
        utt_id,_,spkr,emo_label,sent_label,dia_id,dia_utt_id = row[:7] 
        spkr_id = spk_dict.get(spkr, -1)
        espnet_utt_id = "spk{}-utt{:05d}".format(spkr_id,int(utt_id))
        audio_file_root = f"{audio_dir}/dia{dia_id}_utt{dia_utt_id}"
        wav_scp.append(f"{espnet_utt_id} ffmpeg -i {audio_file_root}.mp4 -f wav pipe:1 |")
        utt2spk.append(f"{espnet_utt_id} {spkr_id}")
        text.append(f"{espnet_utt_id} {emo_label}") 
  
  with open(os.path.join(data_dir,"wav.scp"),"w") as f:
      f.write("\n".join(wav_scp))
  with open(os.path.join(data_dir,"text"),"w") as f:
      f.write("\n".join(text))
  with open(os.path.join(data_dir,"utt2spk"),"w") as f:
      f.write("\n".join(utt2spk))



if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Parse MELD csv's for ESPnet recipe"
  ) 
  parser.add_argument("--label_file")
  parser.add_argument("--audio_dir")
  parser.add_argument("--data_dir")
  args = parser.parse_args()
  parse_csv(
    args.label_file, 
    args.audio_dir, 
    args.data_dir
  )
