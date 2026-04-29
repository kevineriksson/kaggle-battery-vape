import argparse
import json
from pathlib import Path
import pandas as pd
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='Path to best.pt')
    p.add_argument('--test_dir', required=True, help='Folder with test images')
    p.add_argument('--out', required=True, help='Output submission.csv')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--imgsz', type=int, default=640)
    args = p.parse_args()

    class_names = ['battery', 'ecigarette']  # must match training order

    model = YOLO(args.ckpt)
    test_images = sorted(Path(args.test_dir).glob('*.jpg'))

    rows = []
    for img_path in test_images:
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
        )[0]

        preds = []
        if results.boxes is not None:
            for box in results.boxes:
                xyxy = box.xyxy[0].tolist()  # [xmin, ymin, xmax, ymax]
                cls_idx = int(box.cls[0])
                conf = float(box.conf[0])
                preds.append({
                    'class_name': class_names[cls_idx],
                    'confidence': round(conf, 4),
                    'xmin': int(xyxy[0]),
                    'ymin': int(xyxy[1]),
                    'xmax': int(xyxy[2]),
                    'ymax': int(xyxy[3]),
                })

        rows.append({
            'image_id': img_path.name,
            'predictions': json.dumps(preds),
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} predictions to {args.out}")

if __name__ == '__main__':
    main()