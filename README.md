# DensE

## model list : dense, shared, multilayer

### python main.py --model "model" --data "dataset_name" --name "name"

### download trained models : https://drive.google.com/drive/folders/1WY5UzJ08bgEPWFXqxFtjDLHin3KN62V8?usp=sharing

## evaluation)
### python main.py --model dense --width 2 --restore --epoch 0 --name DensE_FB15k-237 --data FB15k-237
### python main.py --model dense --width 100 --restore --epoch 0 --name DensE_WN18RR --data WN18RR

## train example)
### python main.py --model dense --width 2 --data FB15k-237 --name DensE_FB15k-237
### python main.py --model shared --width 100 --data WN18RR --name SharedDensE_100_WN18RR
### python main.py --model multilayer --depth 4 --data FB15k-237 --name MultiLayer_4_FB15k-237
