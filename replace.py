import json

# Buka file json Anda
with open('dataset/udtiri/annotations/instances_train.json', 'r') as f:
    data = json.load(f)

# Ganti nilai "segmentation" menjadi []
for item in data['annotations']:
    if 'segmentation' in item:
        item['segmentation'] = []

# Simpan perubahan ke file json
with open('output/instances_train.json', 'w') as f:
    json.dump(data, f, indent=4)
