# CRF Dialogue Act Classifier
A module for dialogue act tagging.

# Setup
- install the requirments: `requirements.txt`
- Download the trained dact model:


    `wget https://storage.googleapis.com/my_model_files/da-5_acc79.88_loss0.57_e4.pt`

# Using a Trained DACT Model
The file `run.sh` contains an example of how to run dialogue act tagging on the 
example conversations contained within `demo-data.json` and save the results 
in `demo-data-tagged.json` :
```shell
python run.py \
  --conversation_file demo-data.json \
  --model_path da-5_acc79.88_loss0.57_e4.pt \
  --output_path  demo-data-tagged.json \
  --lower \
```