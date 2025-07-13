from PIL import ImageOps
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img # pyright: ignore[reportMissingImports]

# Set the ImageDataGenerator parameters
datagen = ImageDataGenerator(
    rotation_range = 10,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    brightness_range = [0.1, 0.9],
    shear_range = 0.2,
    zoom_range = 0.1,
    channel_shift_range = 10,
    fill_mode = 'nearest'
)


IMG_DIR = "numbers_samples"
PADDING_SIZE = 0.2
EXTRA_IMAGES = 99


def add_padding(img, PADDING_SIZE = 0.2):
    """
    Adds padding to the images to prevent contents 
    from falling out of the frame
    """
    w, h = img.size
    padding_h = int(h * PADDING_SIZE)
    padding_w = int(w * PADDING_SIZE)
    padded_img = ImageOps.expand(
        img, border=(
            padding_w, padding_h, 
            padding_w, padding_h
        ), 
        fill='white'
    )
    return padded_img


counter = 0

# Loop through folder within IMG_DIR
for img_folder in os.listdir(IMG_DIR):
    current_dir = os.path.join(IMG_DIR, img_folder)

    # Loop throguh all images in current_dir
    for img_file in os.listdir(current_dir):

        # Load and modify the image
        img = load_img(os.path.join(current_dir, img_file))
        padded_img = add_padding(img, PADDING_SIZE = 0.2)
        img_array = img_to_array(padded_img) 
        img_array = img_array.reshape((1,) + img_array.shape) # Add a new 'batch' dimension of 1

        for batch in range(0, EXTRA_IMAGES):
            
            # Create a random combination of flow parameters
            flow = datagen.flow(
                img_array, 
                batch_size = 1,
                save_to_dir = current_dir, 
                save_prefix = str(counter),
                save_format = 'png'
            )
            next(flow)
            counter += 1





