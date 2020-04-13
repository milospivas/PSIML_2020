"""
    Task D - WORLD MAP

    Given a "large" image and a set of smaller patches (of constant dimensions)
    find x,y coordinates of each smaller patch in the large image.
    Patches can be exact copies or filtered in some way.
"""

"""
    I've reread the statement and realized that it was guaranteed for the worlmap to remain the same on the private test set.
    Possible solution:
        Use a DNN to 'memorize' the worldmap i.e. to memorize each worlmap's patch
    Outline:
        - train a DNN with
            inputs: the given test set patches - RGB images
            labels: the given test set X,Y coordinate solutions - two nonnegative integers

            - input images aren't all the same dimensions
                - try upscaling to largest / downscaling to smallest / rescaling to some other fixed size

            - ReLU should be fine activation for all layers, including the outputs, since they are nonnegative numbers

            - start with 1 or 2 layers, upscale and optimize along the way

            - there seems to be enough examples for simple 60/20/20 train-CV-test split, if not, use k-fold CV
"""