import os
import glob
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
import cv2

# ==========================================
# AUGMENTATION CONFIGURATION
# Configurable dictionary of probabilities and intensities.
# Tune these after visually inspecting the output.
# ==========================================
AUG_CONFIG = {
    # Standard transforms
    'p_horizontal_flip': 0.5,
    'p_vertical_flip': 0.2,
    'rotation_limit': 30,           # degrees
    'p_rotation': 0.5,
    'scale_limit': 0.2,             # Zoom in/out by 20%
    'p_scale': 0.5,
    'brightness_limit': 0.3,        # Bumped 0.2->0.3 so the effect is visibly present
    'contrast_limit': 0.3,          # Bumped 0.2->0.3 for visibility
    'p_brightness_contrast': 0.8,   # Bumped 0.5->0.8: most samples should shift brightness/contrast

    # Realism-specific transforms
    'p_motion_blur': 0.5,           # Bumped 0.2->0.5
    'blur_limit': (3, 7),           # Kernel size range
    'p_jpeg_compression': 0.6,      # Bumped 0.2->0.6
    'quality_lower': 40,            # Lowered 60->40 so compression artifacts are actually visible
    'quality_upper': 90,            # Lowered 100->90 (100 = near-lossless, no visible change)
    'p_color_temp': 0.7,            # Bumped 0.3->0.7: warm/cool color shifts on most samples
    'color_jitter_hue': 0.15,       # Bumped 0.1->0.15 for a more visible temperature shift
    'color_jitter_saturation': 0.3, # Bumped 0.2->0.3
    'color_temp_limit': (-4000, 4000), # Kelvin-ish shift (simulating warm/cool lighting)

    # Occlusion/Shadow (Random Erasing)
    'p_cutout': 0.4,
    'max_holes': 4,
    'max_height': 32,               # Max pixel height of cutout
    'max_width': 32,                # Max pixel width of cutout

    # Synthetic background paste (highest-priority realism aug: lab -> real farm bg)
    'p_background_paste': 0.4,
}

# Image parameters
IMG_SIZE = 224

# Directory holding background texture images (soil, grass, wood, hand/skin).
# Auto-populated with procedurally generated textures on first use if empty.
BACKGROUNDS_DIR = os.path.join('data', 'backgrounds')
N_BACKGROUNDS = 12

# NOTE: Class Imbalance
# As noted, the dataset has a significant class imbalance 
# (e.g., Potato_healthy: 152 vs Tomato_Yellow_Leaf_Curl_Virus: 5357 images).
# We are NOT handling this via oversampling here. This will be addressed 
# later via class weights in the model training script (model.fit(class_weight=...)).

# ==========================================
# SYNTHETIC BACKGROUND PASTE
# Closes the gap between clean lab photos and real farm photos by isolating
# the leaf and compositing it onto a soil / grass / wood / hand texture.
# This is a deliberately rough first pass (GrabCut + feathered alpha blend).
# ==========================================

# Procedural background palettes (RGB base colors). We generate textures rather
# than requiring a network download so the pipeline is self-contained.
_BG_PALETTES = {
    'soil':  (96, 68, 44),
    'grass': (74, 110, 44),
    'wood':  (150, 108, 66),
    'hand':  (198, 152, 122),   # skin tone
}


def _make_texture(kind, size, seed):
    """Generate a single rough RGB texture (uint8) for the given surface kind."""
    rng = np.random.default_rng(seed)
    base = np.array(_BG_PALETTES[kind], dtype=np.float32)
    img = np.ones((size, size, 3), dtype=np.float32) * base

    # Fine grain noise common to all surfaces
    img += rng.normal(0, 12, (size, size, 3))

    if kind == 'soil':
        # Clumpy speckles: blur coarse noise then add dark/light grains
        coarse = rng.normal(0, 1, (size, size)).astype(np.float32)
        coarse = cv2.GaussianBlur(coarse, (0, 0), sigmaX=4) * 40
        img += coarse[..., None]
    elif kind == 'grass':
        # Vertical blade-like streaks
        streaks = rng.normal(0, 1, (1, size)).astype(np.float32)
        streaks = np.repeat(streaks, size, axis=0)
        streaks = cv2.GaussianBlur(streaks, (3, 0), sigmaX=0.6) * 45
        img += streaks[..., None]
        img[..., 1] += 15  # push green channel
    elif kind == 'wood':
        # Horizontal grain via a sinusoid down the rows
        rows = np.arange(size, dtype=np.float32)
        grain = (np.sin(rows / size * np.pi * rng.uniform(6, 12)) * 22)
        img += grain[:, None, None]
    elif kind == 'hand':
        # Smooth low-frequency lighting gradient over skin
        grad = rng.normal(0, 1, (8, 8, 1)).astype(np.float32)
        grad = cv2.resize(grad, (size, size), interpolation=cv2.INTER_CUBIC)[..., None] * 18
        img += grad

    return np.clip(img, 0, 255).astype(np.uint8)


def generate_background_textures(out_dir=BACKGROUNDS_DIR, n=N_BACKGROUNDS, size=256):
    """
    Generate ~n rough background textures (cycling through soil/grass/wood/hand)
    and save them to out_dir as PNGs. Idempotent-ish: only writes missing files.
    """
    os.makedirs(out_dir, exist_ok=True)
    kinds = list(_BG_PALETTES.keys())
    paths = []
    for i in range(n):
        kind = kinds[i % len(kinds)]
        path = os.path.join(out_dir, f"bg_{i:02d}_{kind}.png")
        if not os.path.exists(path):
            tex = _make_texture(kind, size, seed=i)
            # cv2 writes BGR; our textures are RGB
            cv2.imwrite(path, cv2.cvtColor(tex, cv2.COLOR_RGB2BGR))
        paths.append(path)
    return paths


# Lazy in-memory cache of loaded backgrounds, keyed by directory.
_BG_CACHE = {}


def _get_backgrounds(bg_dir=BACKGROUNDS_DIR):
    """Load (and cache) all background textures as RGB uint8 arrays."""
    if bg_dir in _BG_CACHE:
        return _BG_CACHE[bg_dir]

    files = sorted(glob.glob(os.path.join(bg_dir, '*.png')) +
                   glob.glob(os.path.join(bg_dir, '*.jpg')) +
                   glob.glob(os.path.join(bg_dir, '*.jpeg')))
    if not files:
        # Nothing there yet -> generate the default procedural set
        files = generate_background_textures(bg_dir)

    backgrounds = []
    for f in files:
        img = cv2.imread(f)
        if img is not None:
            backgrounds.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    _BG_CACHE[bg_dir] = backgrounds
    return backgrounds


# GrabCut's cost scales with pixel count, so segment on a downsized copy and
# upscale the resulting alpha mask. 128px keeps mask quality nearly identical
# at ~1/3 the pixels of a 224px input (and far less for larger inputs).
SEGMENT_SIZE = 128


def _segment_leaf(image):
    """
    Rough foreground (leaf) alpha mask for a PlantVillage-style image on a
    mostly-uniform background. Uses GrabCut with a central rectangle, falling
    back to Otsu saturation thresholding if GrabCut degenerates. Segmentation
    runs on a SEGMENT_SIZE x SEGMENT_SIZE copy for speed; the feathered alpha
    is upscaled back. Returns a float alpha map in [0, 1] with shape (H, W).
    """
    image = np.ascontiguousarray(image)
    full_h, full_w = image.shape[:2]

    # Downsize for segmentation — GrabCut runtime scales with pixel count.
    small = cv2.resize(image, (SEGMENT_SIZE, SEGMENT_SIZE),
                       interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]

    mask = None
    try:
        gc_mask = np.zeros((h, w), np.uint8)
        rect = (int(w * 0.08), int(h * 0.08), int(w * 0.84), int(h * 0.84))
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        cv2.grabCut(small, gc_mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        mask = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)
    except Exception:
        mask = None

    frac = float(mask.mean()) if mask is not None else 0.0
    if mask is None or frac < 0.05 or frac > 0.98:
        # Fallback: leaves are more saturated than the neutral lab background
        hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        _, mask = cv2.threshold(sat, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = mask.astype(np.uint8)

    # Clean up: open/close, then keep the largest connected component
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = (labels == largest).astype(np.uint8)

    # Feather the edges so the composite doesn't have a hard cut-out look
    alpha = cv2.GaussianBlur(mask.astype(np.float32), (7, 7), 0)

    # Upscale the alpha back to the source resolution (bilinear keeps the
    # feathered edge smooth).
    if (full_h, full_w) != (h, w):
        alpha = cv2.resize(alpha, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
    return np.clip(alpha, 0.0, 1.0)


def _composite_on_background(image, background, alpha):
    """Alpha-blend the leaf (image) over a background, matched to image size."""
    h, w = image.shape[:2]
    bg = cv2.resize(background, (w, h), interpolation=cv2.INTER_LINEAR)
    a = alpha[..., None]
    out = image.astype(np.float32) * a + bg.astype(np.float32) * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


class BackgroundReplace(A.ImageOnlyTransform):
    """
    Albumentations transform: isolate the leaf and paste it onto a random
    real-world texture. Rough but effective for lab->field domain shift.
    """

    def __init__(self, backgrounds_dir=BACKGROUNDS_DIR, p=0.4):
        super().__init__(p=p)
        self.backgrounds_dir = backgrounds_dir

    def apply(self, img, **params):
        backgrounds = _get_backgrounds(self.backgrounds_dir)
        if not backgrounds:
            return img
        bg = backgrounds[random.randrange(len(backgrounds))]
        alpha = _segment_leaf(img)
        return _composite_on_background(img, bg, alpha)

    def get_transform_init_args_names(self):
        return ('backgrounds_dir',)


# ==========================================
# ALBUMENTATIONS PIPELINE
# ==========================================
def get_training_augmentation(config=AUG_CONFIG):
    """
    Returns an Albumentations composition for training data.
    """
    return A.Compose([
        # Synthetic background paste FIRST, so subsequent geometric / lighting /
        # blur / compression effects unify the pasted leaf with its new background.
        BackgroundReplace(
            backgrounds_dir=BACKGROUNDS_DIR,
            p=config['p_background_paste']
        ),

        # Standard augmentations
        A.HorizontalFlip(p=config['p_horizontal_flip']),
        A.VerticalFlip(p=config['p_vertical_flip']),
        
        # Shift, Scale, Rotate
        A.ShiftScaleRotate(
            shift_limit=0.0, # Handled by other transforms if needed
            scale_limit=config['scale_limit'], 
            rotate_limit=config['rotation_limit'], 
            border_mode=cv2.BORDER_REFLECT_101,
            p=config['p_rotation']
        ),
        
        # Color jitter (Brightness and Contrast)
        A.RandomBrightnessContrast(
            brightness_limit=config['brightness_limit'],
            contrast_limit=config['contrast_limit'],
            p=config['p_brightness_contrast']
        ),
        
        # Realism: Camera/Environment simulation
        A.MotionBlur(blur_limit=config['blur_limit'], p=config['p_motion_blur']),
        
        A.ImageCompression(
            quality_range=(config['quality_lower'], config['quality_upper']),
            p=config['p_jpeg_compression']
        ),
        
        # Color temperature (simulates varying white balance from different cameras/sunlight)
        # RandomColor adjusts HSV values which acts as a proxy for color temperature shifts
        A.ColorJitter(
            hue=config['color_jitter_hue'],
            saturation=config['color_jitter_saturation'],
            p=config['p_color_temp']
        ),
        
        # Realism: Shadows and occlusions
        A.CoarseDropout(
            num_holes_range=(1, config['max_holes']),
            hole_height_range=(8, config['max_height']),
            hole_width_range=(8, config['max_width']),
            fill=0, # Black patches
            fill_mask=None,
            p=config['p_cutout']
        )
    ])

# Initialize the pipeline
train_transform = get_training_augmentation()

# ==========================================
# TF.DATA INTEGRATION
# ==========================================
def _albumentations_wrapper(image, label):
    """
    Applies albumentations pipeline to a single image.
    Expects image in [0, 1] float32 format as output by data_pipeline.py.
    """
    # Albumentations works best with uint8 [0, 255] images
    # We must convert float32 [0.0, 1.0] back to uint8 [0, 255] for Albumentations
    image_np = (image * 255.0).astype(np.uint8)
    
    # Apply transformation
    augmented = train_transform(image=image_np)
    aug_image = augmented['image']
    
    # Convert back to float32 [0.0, 1.0]
    aug_image = (aug_image / 255.0).astype(np.float32)
    
    return aug_image, label

def apply_augmentation(image, label):
    """
    TensorFlow wrapper around the albumentations logic using tf.numpy_function.
    """
    aug_image, label = tf.numpy_function(
        func=_albumentations_wrapper,
        inp=[image, label],
        Tout=[tf.float32, tf.int32] # Assume label is int32 based on data_pipeline.py
    )
    
    # tf.numpy_function loses shape information, so we explicitly set it
    aug_image.set_shape([IMG_SIZE, IMG_SIZE, 3])
    # The shape of the label depends on your pipeline. Usually it's a scalar.
    label.set_shape([])
    
    return aug_image, label

def apply_augmentations_to_dataset(dataset):
    """
    Takes a tf.data.Dataset (training split) and maps the augmentation function.
    """
    # NOTE: Val/test data remains completely unaugmented. Do not pass them here.
    return dataset.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)

# ==========================================
# VISUALIZATION HELPER
# ==========================================
def visualize_augmentations(image_path, n_rows=3, n_cols=3, config=AUG_CONFIG,
                            output_path=os.path.join('outputs', 'aug_preview.png')):
    """
    Plots a grid showing the same source image with different random augmentations applied.
    Useful for visually sanity-checking the intensity of the augmentation pipeline.

    The grid is saved to `output_path` (default: outputs/aug_preview.png) so it works
    from a plain Python terminal without a display / interactive backend.
    """
    # Load raw image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image at {image_path}")
        return
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    # Initialize a new pipeline for visualization to ensure randomness on each call
    transform = get_training_augmentation(config)
    
    plt.figure(figsize=(10, 10))
    plt.suptitle("Augmentation Sanity Check (Center: Original, Others: Augmented)", fontsize=16)
    
    for i in range(n_rows * n_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Center image is original
        if i == (n_rows * n_cols) // 2:
            plt.imshow(image)
            plt.title("Original")
        else:
            augmented = transform(image=image)
            plt.imshow(augmented['image'])
            plt.title(f"Augmented {i}")
            
        plt.axis('off')
        
    plt.tight_layout()

    # Save to file instead of plt.show() so it works from a plain terminal
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Augmentation preview saved to: {os.path.abspath(output_path)}")

if __name__ == '__main__':
    # Simple test block (won't execute fully unless you supply a valid image_path)
    print("Augmentation script loaded successfully.")
    print("To visualize augmentations, run the following in an interactive session:")
    print(">>> from src.augmentation import visualize_augmentations")
    print(">>> visualize_augmentations('path/to/some/plantvillage/image.jpg')")
