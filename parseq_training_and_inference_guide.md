# Custom testing commands for parseq text recognition

## Run the model training (finetune)

Run these in **/str/parseq**

### for SoccerNet data

- **batch size** = 128
- **max epochs** = 25
- **precision** = 16 bit
- **data location** = \data\SoccerNet

```
python train.py +experiment=parseq dataset=real data.root_dir=C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\data\SoccerNet\lmdb trainer.max_epochs=25 pretrained=parseq trainer.accelerator=gpu trainer.devices=1 trainer.val_check_interval=1 data.batch_size=128 data.max_label_length=25 +trainer.precision=16
```

### for custom synthetic data

- **batch size** = 32
- **max epochs** = 10
- **precision** = 64 bit
- **data location** = \dataSynthetic\SyntheticJerseysLarge

```
python train.py +experiment=parseq dataset=real data.root_dir=C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\dataSynthetic\SyntheticJerseysLarge\lmbd trainer.max_epochs=10 pretrained=parseq trainer.accelerator=gpu trainer.devices=1 trainer.val_check_interval=1 data.batch_size=32 data.max_label_length=25 +trainer.precision=64
```

## Run the model inference

If simply evaluating a finetuned model for results, you can run (**in project root**):

- **variable inference_json_path** = the full desired path to the json file that will store str jersey inference results
- **variable parseq_str_checkpoint_path** = the full path to the parseq checkpoint that will be used for str

```
python main.py SoccerNetFinetuned test --jersey_id_result_path %inference_json_path% --str_checkpoint_path %parseq_str_checkpoint_path%
```

Or, if you only want to inference with no need to get results yet, run this in **the project root**

- **fill in variable finetune_folder_name**
- **fill in variable inference_output_results**

```
python str.py C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\str\parseq\outputs\parseq\%finetune_folder_name%\checkpoints\last.ckpt --data_root=C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\out\SoccerNetResults\crops --batch_size=1 --inference --result_file C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\out\SoccerNetResults\%inference_output_results%
```
