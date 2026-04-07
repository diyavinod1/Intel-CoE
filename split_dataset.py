import os
import shutil
import random

source_dir = "Student_Projects"
target_dir = "dataset"

classes = [cls for cls in os.listdir(source_dir) 
           if os.path.isdir(os.path.join(source_dir, cls))]

for cls in classes:
    images = os.listdir(os.path.join(source_dir, cls))
    random.shuffle(images)

    train_split = int(0.7 * len(images))
    val_split = int(0.85 * len(images))

    train_imgs = images[:train_split]
    val_imgs = images[train_split:val_split]
    test_imgs = images[val_split:]

    for split, split_imgs in zip(
        ["train", "val", "test"],
        [train_imgs, val_imgs, test_imgs]
    ):
        split_path = os.path.join(target_dir, split, cls.replace(" ", "_"))
        os.makedirs(split_path, exist_ok=True)

        for img in split_imgs:
            src = os.path.join(source_dir, cls, img)
            dst = os.path.join(split_path, img)
            shutil.copy(src, dst)

print("Dataset split completed!")