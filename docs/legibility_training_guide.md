# HOCKEY

## Train hockey legibility classifier

```
python legibility_classifier.py --train --arch resnet34 --sam --data C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\data\Hockey\jersey_number_dataset --trained_model_path C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\experiments\hockey_legibility.pth
```

## Fine-tune PARSeq STR for HOCKEY number recognition (Trained model will be under str/parseq/outputs)

```
python main.py Hockey train --train_str
```

# SOCCERNET

## Train legibility classifier and jersey number recognition for SoccerNet. Need to first generate weakly labelled datasets and then use them to fine-tune. Weak labels are obtained by using models trained on hockey data above.

```
python legibility_classifier.py --finetune --arch resnet34 --sam --data C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\data\SoccerNetLegibility --full_val_dir C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\data\SoccerNetLegibility\val --trained_model_path C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\experiments\hockey_legibility.pth --new_trained_model_path C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\experiments\sn_legibility.pth
```

### ResNet 50

python legibility_classifier.py --finetune --arch resnet50 --sam --data C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\data\SoccerNetLegibility --full_val_dir C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\data\SoccerNetLegibility\val --trained_model_path C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\experiments\hockey_legibility.pth --new_trained_model_path C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\experiments\legibility_resnet50_20ep.pth

## Fine-tune PARSeq on weakly-labelled SoccerNet data:

```
python main.py SoccerNet train --train_str
```
