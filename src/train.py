import argparse
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Path to data.yaml')
    p.add_argument('--out', required=True, help='Output dir for runs')
    p.add_argument('--model', default='yolov8s.pt')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=16)
    args = p.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.out,
        name='train',
        patience=20,
        save=True,
        plots=True,
    )

if __name__ == '__main__':
    main()