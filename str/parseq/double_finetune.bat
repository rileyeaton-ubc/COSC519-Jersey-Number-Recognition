@REM make sure this matches the main.yaml file in parseq/configs
set parseq_output_folder=2025-03-27_double_finetune_2

@REM make sure to update this in the configuration.py file in root
set inference_output_results=jersey_id_results_test_large_doublefinetune_20ep-32bat-16pr_25ep-128bat-16pr.json

set synthetic_data_epochs=20
set soccernet_epochs=25

set synthetic_data_batch_size=32
set soccernet_batch_size=128

set synthetic_data_pretrain_precision=64
set soccernet_pretrain_precision=16

@REM ----------------------------------
@REM BELOW IS THE CODE TO NOT EDIT
set full_inference_json_path=C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\out\SoccerNetFineTunedResults\%inference_output_results%

del %full_inference_json_path%

del /S /Q C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\str\parseq\outputs\parseq\%parseq_output_folder%\*

@REM train on synthetic number data (large)
python train.py +experiment=parseq dataset=real data.root_dir=C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\dataSynthetic\SyntheticJerseysLarge\lmbd trainer.max_epochs=%synthetic_data_epochs% pretrained=parseq trainer.accelerator=gpu trainer.devices=1 trainer.val_check_interval=1 data.batch_size=%synthetic_data_batch_size% data.max_label_length=25 +trainer.precision=%synthetic_data_pretrain_precision%

@REM Uncomment to also train on soccernet data
python train.py +experiment=parseq dataset=real data.root_dir=C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\data\SoccerNet\lmdb trainer.max_epochs=%soccernet_epochs% pretrained=parseq trainer.accelerator=gpu trainer.devices=1 trainer.val_check_interval=1 data.batch_size=%soccernet_batch_size% data.max_label_length=25 +trainer.precision=%soccernet_pretrain_precision%

cd ..\..

@REM inference
python str.py C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\str\parseq\outputs\parseq\%parseq_output_folder%\checkpoints\last-v1.ckpt --data_root=C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\out\SoccerNetResults\crops --batch_size=1 --inference --result_file %full_inference_json_path%

@REM get resulting accuracy
python main.py SoccerNet test

cd str\parseq