import os
import random
import shutil

# source folders (nested folders)
images_path = "image/image"
labels_path = "labels/labels"

# split ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# create destination folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(split, "images"), exist_ok=True)
    os.makedirs(os.path.join(split, "labels"), exist_ok=True)

# get all images
images = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

print("Total images found:", len(images))

# shuffle dataset
random.shuffle(images)

# calculate split sizes
train_size = int(len(images) * train_ratio)
val_size = int(len(images) * val_ratio)

train_images = images[:train_size]
val_images = images[train_size:train_size + val_size]
test_images = images[train_size + val_size:]


def copy_files(image_list, split):
    for img in image_list:
        label = os.path.splitext(img)[0] + ".txt"

        src_img = os.path.join(images_path, img)
        src_lbl = os.path.join(labels_path, label)

        dst_img = os.path.join(split, "images", img)
        dst_lbl = os.path.join(split, "labels", label)

        shutil.copy2(src_img, dst_img)

        if os.path.exists(src_lbl):
            shutil.copy2(src_lbl, dst_lbl)
        else:
            print("⚠ Missing label for:", img)


copy_files(train_images, "train")
copy_files(val_images, "val")
copy_files(test_images, "test")

print("✅ Dataset successfully split and copied!")