# 16:9GAN

## Overview
Prior to 2009, most television was done in 4:3 aspect ratio, opposed to the current standard of 16:9. 16:9GAN takes in old 4:3 animated video and converts it to 16:9 by generating feasible animation for the sides.

## Methodology

### Data
For data, still frame images where taken from the Fullmetal Alchemist (2003) and Fullmetal Alchemist Brotherhood (2009) anime series. Both are adaptations of the same manga and share almost identical visual styles. However, the 2003 version is in a 4:3 aspect ratio and the 2009 version is in 16:9. From here we trained the GANs on the 16:9 frames. These images were cropped to 4:3 and feed to the generator (I'm using the word cropped, but in reality the image still extends the same ammount but the values have all been set to 0). With an image size of 256, this ammounts to 32 pixels on each side being zeroed out. The generated result along with the original 16:9 image were then given the discriminator to try to distinguish between the real and fake 16:9 image. After the GANs were sufficiently trained, we reviewed the generator results on the 2003 series's 4:3 frames.

### Model
* architecture: using a 256 size
* cost functions
   * Image difference
* stoping condition
    * can just be image simmilarity, when that score discontinues to improve. 
* metrics
    * Gen Loss
    * Disc Loss
    * Image simmilarity

<insert image of architecture>
256 x 256 image convolution layers doubling 3 to 24 at 32 pixels.

## Notebooks
* process_freams.ipynb: Takes the video files and converts them to frames. 
* model_image_difference.ipynb: model that uses the MSE of the original image and the generated image as the loss. This method only has a generator and not a discrimiator.

## Results to show
* Results over time
* Example on the 16:9
* Example on the 4:3
    * fixed background vs action sceen
* Look at other animated shows (Simpsons etc)

## Backlog
* try just using a generator and having the image difference be the cost function
* try the GANs approach
* read up on different architecture
