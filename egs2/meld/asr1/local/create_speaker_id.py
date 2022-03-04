from pandas import read_csv 
import argparse
import pdb
import pickle

def create_mapping(f):
  csv = read_csv(f)
  unique_speakers = csv.Speaker.unique()
  spk_dict = dict()
  for i, speaker in enumerate(sorted(unique_speakers)):
    spk_dict[speaker] = i 
  with open('speaker_dict.pkl', 'wb') as f:
    pickle.dump(spk_dict, f)
  return 
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description="Create dictionary for speaker to speaker-id"
  ) 
  parser.add_argument("--fpath")
  args = parser.parse_args()
  create_mapping(args.fpath)
