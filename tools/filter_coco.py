import json
import os

def filter_coco_strict(json_path, output_path, target_ids=[3, 6, 8]):
    print(f"opening {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    new_anns = []
    valid_image_ids = set()
    for ann in data['annotations']:
        if ann['category_id'] in target_ids:
            ann['category_id'] = 0  # Remap to 'vehicle'
            new_anns.append(ann)
            valid_image_ids.add(ann['image_id'])

    new_imgs = []
    print(f"dropping {len(data['images']) - len(valid_image_ids)} empty images")
    for img in data['images']:
        if img['id'] in valid_image_ids:
            new_imgs.append(img)

    data['annotations'] = new_anns
    data['images'] = new_imgs
    data['categories'] = [{"id": 0, "name": "vehicle"}]
    with open(output_path, 'w') as f:
        json.dump(data, f)


if not os.path.exists('../coco/annotations'): os.makedirs('coco/annotations')
filter_coco_strict('../coco/annotations/instances_train2017.json', '../coco/annotations/teacher_train.json')
filter_coco_strict('../coco/annotations/instances_val2017.json', '../coco/annotations/teacher_val.json')