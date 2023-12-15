# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt


# def show_image_blocks(image_path, block_size):
#     # Load the image
#     original_image = Image.open(image_path)

#     # Convert the image to a NumPy array
#     image_array = np.array(original_image)

#     # Get the dimensions of the image
#     height, width, channels = image_array.shape

#     # Calculate the number of blocks in each dimension
#     num_blocks_x = width // block_size
#     num_blocks_y = height // block_size

#     # Set up subplots for displaying blocks
#     fig, axs = plt.subplots(num_blocks_y, num_blocks_x, figsize=(10, 10))

#     # Iterate over the blocks
#     for i in range(num_blocks_y):
#         for j in range(num_blocks_x):
#             # Calculate the slicing coordinates for each block
#             start_x = j * block_size
#             start_y = i * block_size
#             end_x = start_x + block_size
#             end_y = start_y + block_size

#             # Extract the block from the image array
#             block = image_array[start_y:end_y, start_x:end_x, :]

#             # Display the block
#             axs[i, j].imshow(block)
#             axs[i, j].axis("off")

#     plt.show()


