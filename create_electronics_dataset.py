"""
Script untuk membuat dataset komponen elektronik dalam format YOLO
"""

import os
import cv2
import numpy as np
import glob
from pathlib import Path
import shutil
import argparse
import logging

# Konfigurasi logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_yolo_dataset(source_dir, output_dir, split_ratio=0.8):
    """
    Membuat dataset dalam format YOLO dari folder gambar yang berisi subfolder per kelas
    
    Args:
        source_dir: Direktori sumber yang berisi subfolder untuk setiap kelas
        output_dir: Direktori output untuk dataset YOLO
        split_ratio: Rasio pembagian train/val (default: 0.8)
    """
    # Buat struktur folder YOLO
    os.makedirs(output_dir, exist_ok=True)
    
    # Buat subfolder
    images_train_dir = os.path.join(output_dir, "images", "train")
    images_val_dir = os.path.join(output_dir, "images", "val")
    labels_train_dir = os.path.join(output_dir, "labels", "train")
    labels_val_dir = os.path.join(output_dir, "labels", "val")
    
    os.makedirs(images_train_dir, exist_ok=True)
    os.makedirs(images_val_dir, exist_ok=True)
    os.makedirs(labels_train_dir, exist_ok=True)
    os.makedirs(labels_val_dir, exist_ok=True)
    
    # Buat daftar kelas dari subfolder
    class_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    class_dirs.sort()  # Pastikan urutan sama setiap kali
    
    logger.info(f"Kelas yang ditemukan: {class_dirs}")
    
    # Buat file data.yaml
    with open(os.path.join(output_dir, "data.yaml"), 'w') as f:
        f.write(f"path: {output_dir}\n")
        f.write(f"train: images/train\n")
        f.write(f"val: images/val\n")
        f.write("\n")
        f.write(f"nc: {len(class_dirs)}\n")
        f.write(f"names: {class_dirs}\n")
    
    # Proses setiap kelas
    for class_id, class_name in enumerate(class_dirs):
        class_dir = os.path.join(source_dir, class_name)
        logger.info(f"Memproses kelas {class_name} (ID: {class_id})")
        
        # Dapatkan semua gambar
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(glob.glob(os.path.join(class_dir, ext)))
            image_files.extend(glob.glob(os.path.join(class_dir, ext.upper())))
        
        logger.info(f"  - Menemukan {len(image_files)} gambar")
        
        # Acak dan bagi menjadi train/val
        np.random.shuffle(image_files)
        n_train = int(len(image_files) * split_ratio)
        
        train_files = image_files[:n_train]
        val_files = image_files[n_train:]
        
        logger.info(f"  - Train: {len(train_files)}, Val: {len(val_files)}")
        
        # Proses gambar train
        for img_path in train_files:
            process_image(img_path, images_train_dir, labels_train_dir, class_id)
        
        # Proses gambar val
        for img_path in val_files:
            process_image(img_path, images_val_dir, labels_val_dir, class_id)
    
    logger.info(f"Dataset komponen elektronik berhasil dibuat di {output_dir}")
    logger.info(f"Total gambar train: {len(os.listdir(images_train_dir))}")
    logger.info(f"Total gambar val: {len(os.listdir(images_val_dir))}")
    logger.info("\nPenggunaan:")
    logger.info(f"1. Pastikan ultralytics sudah terinstall: pip install ultralytics")
    logger.info(f"2. Latih model: yolo train data={os.path.join(output_dir, 'data.yaml')} model=yolov8n.pt epochs=50")

def process_image(img_path, images_dir, labels_dir, class_id):
    """
    Memproses gambar dan membuat anotasi YOLO
    
    Args:
        img_path: Path ke file gambar
        images_dir: Direktori untuk menyimpan gambar
        labels_dir: Direktori untuk menyimpan label
        class_id: ID kelas untuk objek
    """
    try:
        # Baca gambar untuk mendapatkan dimensi
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Tidak bisa membaca gambar: {img_path}")
            return
            
        height, width = img.shape[:2]
        
        # Buat nama file baru
        img_filename = Path(img_path).name
        base_name = Path(img_filename).stem
        
        # Salin gambar ke folder tujuan
        dst_img_path = os.path.join(images_dir, img_filename)
        shutil.copy2(img_path, dst_img_path)
        
        # Di sini kita akan mengasumsikan bahwa objek mengisi sebagian besar gambar
        # Untuk dataset sesungguhnya, Anda perlu anotasi yang tepat per objek
        # Asumsi objek ada di tengah dan mengisi ~80% gambar
        center_x = 0.5  # Relatif terhadap width
        center_y = 0.5  # Relatif terhadap height
        bbox_width = 0.8  # Relatif terhadap width
        bbox_height = 0.8  # Relatif terhadap height
        
        # Tulis file label YOLO (format: class_id center_x center_y width height)
        label_path = os.path.join(labels_dir, f"{base_name}.txt")
        with open(label_path, 'w') as f:
            f.write(f"{class_id} {center_x} {center_y} {bbox_width} {bbox_height}\n")
            
    except Exception as e:
        logger.error(f"Error memproses gambar {img_path}: {str(e)}")

def interactive_labeling(source_dir, output_dir):
    """
    Alat interaktif untuk membuat label YOLO dengan bantuan pengguna
    
    Args:
        source_dir: Direktori sumber dengan struktur kelas/gambar
        output_dir: Direktori output untuk dataset YOLO
    """
    logger.info("Mode interaktif belum diimplementasikan - akan menggunakan mode otomatis")
    create_yolo_dataset(source_dir, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Membuat dataset komponen elektronik dalam format YOLO")
    parser.add_argument("--source", required=True, help="Direktori sumber yang berisi subfolder untuk setiap kelas")
    parser.add_argument("--output", required=True, help="Direktori output untuk dataset YOLO")
    parser.add_argument("--interactive", action="store_true", help="Gunakan mode interaktif untuk membuat label")
    parser.add_argument("--split", type=float, default=0.8, help="Rasio pembagian train/val (default: 0.8)")
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_labeling(args.source, args.output)
    else:
        create_yolo_dataset(args.source, args.output, args.split)