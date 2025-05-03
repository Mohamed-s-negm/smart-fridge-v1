import os
import shutil

def prepare_food101_dataset(root_dir='../datasets/food-101-dataset/food-101'):
    images_dir = os.path.join(root_dir, 'images')
    meta_dir = os.path.join(root_dir, 'meta')
    
    train_dir = os.path.join(root_dir, 'train')
    val_dir = os.path.join(root_dir, 'val')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Helper to move files
    def move_images(txt_file, dest_dir):
        with open(os.path.join(meta_dir, txt_file), 'r') as f:
            lines = f.readlines()
        for line in lines:
            class_name, image_file = line.strip().split('/')
            src = os.path.join(images_dir, class_name, image_file + '.jpg')
            class_dest = os.path.join(dest_dir, class_name)
            os.makedirs(class_dest, exist_ok=True)
            shutil.copy(src, os.path.join(class_dest, image_file + '.jpg'))

    print("Copying training images...")
    move_images('train.txt', train_dir)

    print("Copying validation images...")
    move_images('test.txt', val_dir)

    print("✅ Dataset prepared in 'train/' and 'val/' folders.")

prepare_food101_dataset()
