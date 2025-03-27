import json
import random

json_path = 'dataSynthetic\\SyntheticJerseysLarge\\all\\groundtruth.json'
images_folder = 'C:\\Users\\Riley\\Documents\\UBC\\GitHub\\COSC519-Jersey-Number-Recognition\\dataSynthetic\\SyntheticJerseysLarge\\all\\image'
output_txt_path = 'dataSynthetic\\SyntheticJerseysLarge'
image_type = "png"

with open(json_path, 'r', encoding='utf-8') as f:
  data = json.load(f)

items = list(data.items())  # [(img_name, label), (img_name, label), ...]
random.shuffle(items)       # in-place shuffle

total_len = len(items)
train_end = int(0.7 * total_len)
val_end = int(0.9 * total_len)

train_items = items[:train_end]
val_items   = items[train_end:val_end]
test_items  = items[val_end:]

def write_gt_file(split_items, out_txt):
  with open(out_txt, 'w', encoding='utf-8') as out:
    for img_name, label in split_items:
      full_img_path = f"{images_folder}\\{img_name}.{image_type}"
      out.write(f"{full_img_path}\t{label}\n")

write_gt_file(train_items, f"{output_txt_path}\\gt_train.txt")
write_gt_file(val_items,   f"{output_txt_path}\\gt_val.txt")
write_gt_file(test_items,  f"{output_txt_path}\\gt_test.txt")

print(f"Saved ground-truth files")
