# DensE

## model list : dense, multilayer, shared

### python main.py --model "model_name" --data "dataset_name"

### download trained models : https://drive.google.com/drive/folders/1WY5UzJ08bgEPWFXqxFtjDLHin3KN62V8?usp=sharing

## evaluation)
### python main.py --model dense --heads 2 --restore --epoch 0 --name DensE_FB15k-237 --data FB15k-237
### python main.py --model dense --heads 100 --restore --epoch 0 --name DensE_FB15k-237 --data WN18RR

## train example)
### python main.py --model dense --data FB15k-237
### python main.py --model shared --heads 100 --data WN18RR
### python main.py --model multilayer --layers 4 --data FB15k-237
### python main.py --model dense --data FB15k-237 --operation add
