import os
import random
import shutil

random.seed(42)

train_dir = "dataset/train"
val_dir = "dataset/validation"

split_ratio = 0.2  # 20% untuk validation

os.makedirs(val_dir, exist_ok=True)

classes = os.listdir(train_dir)

for cls in classes:
    train_class = os.path.join(train_dir, cls)
    val_class = os.path.join(val_dir, cls)

    os.makedirs(val_class, exist_ok=True)

    images = os.listdir(train_class)
    random.shuffle(images)

    num_val = int(len(images) * split_ratio)

    val_images = images[:num_val]

    for img in val_images:
        src = os.path.join(train_class, img)
        dst = os.path.join(val_class, img)
        shutil.move(src, dst)

print("Validation dataset berhasil dibuat!")