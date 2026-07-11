import os
import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Configuration constants
DATA_DIR = './data/plantvillage_dataset/color/'
TARGET_CROPS = ('Tomato', 'Potato', 'Corn')
CSV_PATH = './data/dataset_split.csv'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

def create_dataset_csv(data_dir=DATA_DIR, output_csv=CSV_PATH):
    """
    Scans the data directory, filters for specified crops, prints class stats,
    and creates a stratified 70/15/15 train/val/test split saved to a CSV.
    """
    image_paths = []
    labels = []
    class_counts = {}

    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        print("Please ensure the dataset is located at the specified path.")
        return None, None
        
    for class_folder in os.listdir(data_dir):
        if class_folder.startswith(TARGET_CROPS):
            folder_path = os.path.join(data_dir, class_folder)
            if os.path.isdir(folder_path):
                # Discover images
                images = glob.glob(os.path.join(folder_path, '*.*'))
                # Filter to standard image extensions just in case
                images = [img for img in images if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if not images:
                    continue
                    
                class_counts[class_folder] = len(images)
                
                # Use normalized paths to ensure consistency
                normalized_images = [os.path.normpath(img) for img in images]
                image_paths.extend(normalized_images)
                labels.extend([class_folder] * len(images))

    if not image_paths:
        print("No images found for the specified crops.")
        return None, None

    # 1. Print class names and image counts per class
    print("Found the following classes and image counts:")
    for cls, count in class_counts.items():
        print(f" - {cls}: {count} images")
    print(f"Total images: {len(image_paths)}")

    # 2. Create stratified splits (70% train, 15% val, 15% test)
    # First split: 70% train, 30% temp (which will be divided into val and test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        image_paths, labels, test_size=0.30, stratify=labels, random_state=42
    )
    
    # Second split: divide temp equally into 50% val and 50% test (15% overall each)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    # Combine into a DataFrame
    df_train = pd.DataFrame({'image_path': X_train, 'label': y_train, 'split': 'train'})
    df_val = pd.DataFrame({'image_path': X_val, 'label': y_val, 'split': 'val'})
    df_test = pd.DataFrame({'image_path': X_test, 'label': y_test, 'split': 'test'})
    
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)
    
    # 3. Save the split as a CSV for reproducibility
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_all.to_csv(output_csv, index=False)
    print(f"\nSaved dataset split to {output_csv}")
    
    # Create label mapping to integer for tf.data
    unique_labels = sorted(list(set(labels)))
    label_to_index = {label: i for i, label in enumerate(unique_labels)}
    
    return df_all, label_to_index

def load_and_preprocess_image(image_path, label):
    """
    Reads an image from disk, decodes, resizes, and normalizes it.
    """
    # Read the image file from disk
    img = tf.io.read_file(image_path)
    
    # Decode to RGB (channels=3) - assuming JPEG images which is standard for PlantVillage
    img = tf.image.decode_jpeg(img, channels=3)
    
    # Resize to the target input size for the model
    img = tf.image.resize(img, IMG_SIZE)
    
    # Normalize pixel values to the range [0, 1]
    img = img / 255.0
    
    return img, label

def build_dataset(df, split, label_to_index, batch_size=BATCH_SIZE):
    """
    Builds an optimized tf.data.Dataset from a split dataframe.
    """
    split_df = df[df['split'] == split]
    
    # Map string labels to integers
    int_labels = [label_to_index[label] for label in split_df['label']]
    image_paths = split_df['image_path'].tolist()
    
    # Create dataset from tensor slices
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, int_labels))
    
    # Shuffle only if it's the training set
    if split == 'train':
        dataset = dataset.shuffle(buffer_size=len(image_paths))
        
    # Apply basic preprocessing with parallel execution
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch and prefetch for performance
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset

if __name__ == '__main__':
    print("Initializing dataset pipeline...")
    
    # Execute the CSV creation and split logic
    df, label_map = create_dataset_csv()
    
    if df is not None:
        print("\nLabel to Index Mapping:")
        for label, idx in label_map.items():
            print(f" {idx}: {label}")
            
        print("\nBuilding tf.data.Dataset objects...")
        # 4. Build tf.data.Dataset objects from the CSV
        train_ds = build_dataset(df, 'train', label_map)
        val_ds = build_dataset(df, 'val', label_map)
        test_ds = build_dataset(df, 'test', label_map)
        
        print("\nDataset creation complete!")
        print(f"Train batches: {len(train_ds)}")
        print(f"Val batches: {len(val_ds)}")
        print(f"Test batches: {len(test_ds)}")
