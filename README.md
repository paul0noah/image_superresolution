# image_superresolution
A short python project which allows you to compute a superresolution image out of several, unaligned images of the same subject. 
Take a look at this video 
`https://www.youtube.com/watch?v=2QW9vcnb9c0&t=207s`
to get a better idea whats it about.

## Usage

Either take a look at the jupyter notebook at `src/superresolution.ipynb` or follow these steps



Install the dependencies and devDependencies and start the server.

1) Define a list of filenames (where the filenames are the path to some images)
2) Execute the following code
    ```python
    from src.helper.superresolution_pipeline import *
    superresolution_pipeline(filenames,	output_filename='superres.jpg', resize_scale=5)
    ```
3) Now you have a superresolution image, composed out of your input images saved at `superres.jpg`

## Plugins

We need the following python modules to run the functions correctly
| Plugin | Installation |
| ------ | ------ |
| cv2 | `pip install opencv-python==3.4.2.17` | 
| cv2-contrib | `pip install opencv-contrib-python==3.4.2.17` |
| numpy | `pip install numpy` |


