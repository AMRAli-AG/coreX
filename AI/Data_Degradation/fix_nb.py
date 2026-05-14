import json

with open('ts_augmentation_benchmark.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if '%pip install' in line or '!pip install' in line:
                new_source.append('# ' + line)
            else:
                new_source.append(line)
        cell['source'] = new_source

with open('ts_augmentation_benchmark.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
    
print("Notebook fixed.")
