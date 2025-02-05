file_path = "/network/scratch/s/saba.ahmadi/emu3/checkpoints/Emu3-Base-SFT-VisMin256-CoT-17jan/checkpoint-8008/results_training/48/CoT_visMin_eval_train-CoT_visMin_eval_training_wo_cfg-20250203-131817/eval_report.jsonl"
import pandas as pd
import csv

jsonObj = pd.read_json(path_or_buf=file_path, lines=True)
required_samples = [8, 16, 21, 23, 28]

data = [
    {'idx': idx,
     'edit_instruction': jsonObj['edit_instruction'][idx],
     'orig_image': f'./assets/org_{idx}.png',
     'gt_edit': f'./assets/gt_edited_{idx}.png',
     'our_edit': f'./assets/result_{idx}_1.png'} for idx in required_samples
]

from editscore.baseline_models import InstructPix2Pix
ip2p = InstructPix2Pix()
from PIL import Image
for sample in data:
    orig = Image.open(sample['orig_image'])
    prompt = sample['edit_instruction']
    edited = ip2p.get_editted_image(prompt, orig)
    edited.save(f'./assets/baseline_{sample["idx"]}.png')
    sample['baseline_edit'] = f'./assets/baseline_{sample["idx"]}.png'


field_names = ['idx', 'edit_instruction', 'orig_image', 'gt_edit', 'our_edit', 'baseline_edit']
with open('edit_prompt.csv', 'w') as f:
    writer = csv.DictWriter(f, fieldnames=field_names)
    writer.writeheader()
    writer.writerows(data)