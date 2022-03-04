import csv
import argparse
import pdb
import pickle

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

def parse_csv(label_file, audio_dir, wav_scp_path, utt2spk_path, 
              text_path):
  with open("speaker_dict.pkl", 'rb') as f:
    spk_dict = pickle.load(f)
  [wav_scp, utt2spk, text]  = [
    open(path, "w") for path in [wav_scp_path, utt2spk_path, text_path]
  ]
 
  with open(label_file) as csvfile:
      csv_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
      next(csv_reader, None)  # skip header 
      for row in csv_reader:
        # adding speakerid as prefix of utterance id because of note in kaldi data-prep document: 
        # https://kaldi-asr.org/doc/data_prep.html
        meld_utt_id = row[UTT_ID_i] 
        spkr_id = spk_dict.get(row[SPK_i], -1)
        dia_id = row[DIA_ID_i]
        emo_label = row[EMO_i]
        espnet_utt_id = f"spk{spkr_id}-utt{meld_utt_id}"
        audio_file_root = f"{audio_dir}/dia{dia_id}_utt{meld_utt_id}"

        # write to necessary files
        wav_scp.write(f"{espnet_utt_id} ffmpeg -i {audio_file_root}.mp4 {audio_file_root}.wav\n")
        utt2spk.write(f"{espnet_utt_id} {spkr_id}\n")
        text.write(f"{espnet_utt_id} {emo_label}\n") 
  wav_scp.close(); utt2spk.close(); text.close(); 



if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Parse MELD csv's for ESPnet recipe"
  ) 
  parser.add_argument("--label-file")
  parser.add_argument("--audio-dir")
  parser.add_argument("--wav-scp")
  parser.add_argument("--utt2spk")
  parser.add_argument("--text")
  args = parser.parse_args()
  parse_csv(
    args.label_file, 
    args.audio_dir, 
    args.wav_scp, 
    args.utt2spk,
    args.text,
  )
