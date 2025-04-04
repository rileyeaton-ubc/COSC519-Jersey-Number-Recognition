@REM make sure this matches the main.yaml file in parseq/configs
set parseq_output_folder=2025-03-30_double_finetune_final_synthetic_64bit
set parseq_str_checkpoint_path=C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\str\parseq\outputs\parseq\%parseq_output_folder%\checkpoints\last-v1.ckpt

@REM make sure to update this in the configuration.py file in root
set inference_output_results=synthetic_jersey_id_results_final_doublefinetune_15ep-116bat-64pr_13ep-116bat-64pr.json

set synthetic_data_epochs=15
set synthetic_data_batch_size=116
set synthetic_data_pretrain_precision=64

set soccernet_epochs=20
set soccernet_batch_size=116
set soccernet_pretrain_precision=64

@REM ----------------------------------
@REM BELOW IS THE CODE TO NOT EDIT
set inference_json_path=C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\out\SoccerNetFineTunedResults\%inference_output_results%

del %inference_json_path%

del /S /Q C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\str\parseq\outputs\parseq\%parseq_output_folder%\*

@REM train on synthetic number data (final)
python train.py +experiment=parseq dataset=real data.root_dir=C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\dataSynthetic\SyntheticJerseysFinal\lmbd trainer.max_epochs=%synthetic_data_epochs% pretrained=parseq trainer.accelerator=gpu trainer.devices=1 trainer.val_check_interval=1 data.batch_size=%synthetic_data_batch_size% data.max_label_length=2 +trainer.precision=%synthetic_data_pretrain_precision%

@REM Uncomment to also train on soccernet data
python train.py +experiment=parseq dataset=real data.root_dir=C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\data\SoccerNet\lmdb trainer.max_epochs=%soccernet_epochs% pretrained=parseq trainer.accelerator=gpu trainer.devices=1 trainer.val_check_interval=1 data.batch_size=%soccernet_batch_size% data.max_label_length=2 +trainer.precision=%soccernet_pretrain_precision%

cd ..\..

@REM inference model and get resulting accuracy
python main.py SoccerNetFinetuned test --jersey_id_result_path %inference_json_path% --str_checkpoint_path %parseq_str_checkpoint_path%

cd str\parseq