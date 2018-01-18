readme for hot dog or not a hot dog.


Aim was to build classifier to classify whether an image is a hotdog or not.
Built using a convolutional neural net in Tensorflow. 
Not that many images, around a 100 for each category so not that robust.

The preprocessing file reads in the images and creates the vector representation of each image. 
The vectors are 784 elemtns long as the images are all normalized to 28 * 28 size. The values stored in each element are the 
pixel intensity values(greyscale values) for the respective pixel.
These vectors are stored in a list of size (238, 900)
The label list is also created and is size (238, 1) containing values of 0 (not a hot dog ) or 1 (a hot dog)
These lists are converted to nump arrays, stored in a tuple and saved as a pickle file.

The hotdog file is where the classifier is built.

Firstly read in the pickle file.
One hot encoding is used to convert the label array to (238,2) where the 1st column is the not hot dog column and the 2nd is the hot dog column.
Make a training test split.
