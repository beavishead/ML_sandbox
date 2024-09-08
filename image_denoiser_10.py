import numpy as np
from scipy.ndimage import generic_filter
from PIL import Image

class ImageDenoiser:

    """
    THE DENOISER CLASS. Uploads, restores and saves the highly damaged images
    (up to 85% of pixel values are omitted)
    The main procedure is filling out the omitted values with the random non-zero
    values in the closest neighbourhood, iteratively
    """

    ### initialize the class with reading the data from the numpy dataset
    def __init__(self,file_path):
        with open(file_path,mode='rb') as file:
            self.images = np.load(file)
        self.pics = []

    ### Assign a pixel value to the random non-zero value in its vicinity
    ### If the pixel value is non-zero initially remain it untouched
    ### If there are no non-zero values inside the neighbourhood leave the pixel value 0
    def random_neighbour_fill(self,values):

        center_pixel = values[len(values) // 2]

        if center_pixel != 0:
            return center_pixel

        non_zero_neighbours = values[values != 0]

        if len(non_zero_neighbours) > 0:
            return np.random.choice(non_zero_neighbours)
        else:
            return center_pixel

    ### Using the generic filter from scipy.ndimage fill out the zero values in three channels in a given image
    ### repeat {iterations} number of times, to handle big areas of zero values
    def fill_missing_values_random(self,image,iterations=3):
        filled_image=image.copy()

        for _ in range(iterations):
            for i in range(3):
                channel = filled_image[:, :, i]

                filled_channel=generic_filter(channel,self.random_neighbour_fill,size=3,mode="constant",cval=0.0)

                filled_image[:, :, i] = filled_channel

        return filled_image

    ### apply the filter to every image
    ### check that the colours are not inverted
    ### convert back to 255 for PIL
    def process_images(self, iterations=3):
        for img in range(self.images.shape[0]):
            image = self.images[img]

            filled_image = self.fill_missing_values_random(image,iterations=iterations)

            if np.mean(filled_image) >1.0:
                filled_image = 1.0 - filled_image

            filled_image_uint8 = (filled_image * 255).astype(np.uint8)
            self.pics.append(filled_image_uint8)

    ### show and save the denoised images both as the pictures and as numpy array
    def save_images(self,output_image_path='static/data/denoising/filled_image', output_npy_path="static/data/denoising/denoised_images.npy"):
        iter=0
        for filled_image_uint8 in self.pics:
            extension = '.png'
            img = Image.fromarray(filled_image_uint8, "RGB")
            img.save(f"{output_image_path}_"+str(iter)+extension)
            img.show()
            iter+=1

        with open(output_npy_path, mode='wb') as file:
            np.save(file, np.array(self.pics).astype(np.uint8), allow_pickle=False, fix_imports=False)

# Example of usage
denoiser = ImageDenoiser('static/data/denoising/data.npy')
denoiser.process_images()
denoiser.save_images()


