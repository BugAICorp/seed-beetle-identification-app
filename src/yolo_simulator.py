""" yolo_simulator.py """

from pathlib import Path
from torch.serialization import add_safe_globals
from ultralytics.nn.tasks import DetectionModel
add_safe_globals([DetectionModel])
from ultralytics import YOLO
from yolo_dataset_builder import YoloDatasetBuilder
from yolo_training_program import YOLOTrainer
from globals import yolo_model

if __name__ == "__main__":
    print("Would you like to train a new model? (Press Enter to skip, or type anything to train)")
    choice = input("Choice: ").strip()

    if choice:
        # Build dataset split
        print("\nBuilding dataset split...")
        builder = YoloDatasetBuilder(
            source_dir="dataset",
            output_dir="yolo_dataset",
            train_ratio=0.8
        )
        builder.build(total_images=2000)
        print("Dataset split built!")

        # Train the YOLO model
        print("\nStarting training...")
        trainer = YOLOTrainer(dataset_yaml=builder.yaml, epochs=10, batch_size=8, img_size=512)
        trainer.train()
        trainer.save()
        print("Training complete and model saved!")

    else:
        print("\nSkipping training. Using existing trained model at:", yolo_model)
        builder = None  # Avoid calling cleanup if builder was never initialized

    # Run prediction interactively
    print("\nYou can now test an image.")
    while True:
        img_path = input("Enter the path of an image to test (or 'exit' to quit): ").strip()
        if img_path.lower() == 'exit':
            break
        if not Path(img_path).is_file():
            print("Invalid file path. Try again.")
            continue

        # Load trained model and predict
        model = YOLO(yolo_model)
        results = model(img_path, imgsz=512, conf=0.25)

        # Extract and display bounding box info
        result = results[0]
        boxes = result.boxes  # ultralytics.engine.results.Boxes object

        print(f"\nDetected {len(boxes)} object(s):")
        for i, box in enumerate(boxes):
            xyxy = box.xyxy.cpu().numpy()[0].tolist()  # [x1, y1, x2, y2]
            conf = float(box.conf.cpu().numpy()[0])
            cls = int(box.cls.cpu().numpy()[0])
            print(f"  Box {i + 1}: Class {cls} | Confidence {conf:.2f} | Coordinates: {xyxy}")

        # Draw the boxes on the image and display it
        result.show()

    print("Cleaning up and exiting : )")
    if builder:
        builder.cleanup()
