# CombinE

## model list : combine, shared_only, multilayer(shared_only)

### python main.py --model "model" --data "dataset_name" --name "name"

### download trained models : https://drive.google.com/drive/folders/1GeGdZgnanNYbEwc_rqMQM71Lv16M5F0P?usp=sharing

## evaluation) : download pretrained models from google drive and save to directory ./torch_saved
### python main.py --model combine --width 2 --restore --epoch 0 --name CombinE_FB15k-237 --data FB15k-237
### python main.py --model combine --width 100 --restore --epoch 0 --name CombinE_WN18RR --data WN18RR

## train example)
### python main.py --model combine --width 2 --data FB15k-237 --name CombinE_FB15k-237
### python main.py --model shared --width 100 --data WN18RR --name SharedDensE_100_WN18RR
### python main.py --model multilayer --depth 4 --data FB15k-237 --name MultiLayer_4_FB15k-237
