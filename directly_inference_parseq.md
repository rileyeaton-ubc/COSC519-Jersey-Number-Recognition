# Custom testing results

## Run just the model inference using conda

```
conda run -n parseq2 python str.py C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\models\soccernet-personal\parseq_epoch=22-step=2369-val_accuracy=95.5357-val_NED=96.3599.ckpt --data_root=C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\out\SoccerNetResults\crops --batch_size=1 --inference --result_file C:\Users\Riley\Documents\UBC\GitHub\COSC519-Jersey-Number-Recognition\out\SoccerNetResults\jersey_id_results_test.json
```
