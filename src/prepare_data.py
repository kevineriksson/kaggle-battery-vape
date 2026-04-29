import argparse
import json
import shutil
from pathlib import Path
import random

def coco_to_yolo(coco_json, images_dir, out_root, val_frac=0.2, seed=42):
    with open(coco_json) as f:
        coco = json.load(f)

    # category_id → 0-indexed YOLO class
    cat_id_to_yolo = {c['id']: i for i, c in enumerate(coco['categories'])}
    class_names = [c['name'] for c in coco['categories']]

    # group annotations by image
    img_anns = {img['id']: [] for img in coco['images']}
    for ann in coco['annotations']:
        img_anns[ann['image_id']].append(ann)

    # split images into train/val
    random.seed(seed)
    image_ids = list(img_anns.keys())
    random.shuffle(image_ids)
    n_val = int(len(image_ids) * val_frac)
    val_ids = set(image_ids[:n_val])

    out_root = Path(out_root)
    for split in ['train', 'val']:
        (out_root / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out_root / 'labels' / split).mkdir(parents=True, exist_ok=True)

    img_lookup = {img['id']: img for img in coco['images']}

    for img_id, anns in img_anns.items():
        img = img_lookup[img_id]
        split = 'val' if img_id in val_ids else 'train'
        w, h = img['width'], img['height']

        # copy image
        src_img = Path(images_dir) / img['file_name']
        dst_img = out_root / 'images' / split / img['file_name']
        shutil.copy(src_img, dst_img)

        # write YOLO label
        label_path = out_root / 'labels' / split / (Path(img['file_name']).stem + '.txt')
        with open(label_path, 'w') as f:
            for a in anns:
                x, y, bw, bh = a['bbox']
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h
                cls = cat_id_to_yolo[a['category_id']]
                f.write(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    # write data.yaml
    yaml_text = f"""path: {out_root.absolute()}
train: images/train
val: images/val
nc: {len(class_names)}
names: {class_names}
"""
    (out_root / 'data.yaml').write_text(yaml_text)
    print(f"Wrote {len(image_ids)} images: {len(image_ids) - n_val} train / {n_val} val")
    print(f"Classes: {class_names}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--coco', required=True, help='Path to COCO JSON')
    p.add_argument('--images', required=True, help='Path to image folder')
    p.add_argument('--out', required=True, help='Output YOLO dataset root')
    p.add_argument('--val_frac', type=float, default=0.2)
    args = p.parse_args()
    coco_to_yolo(args.coco, args.images, args.out, args.val_frac)