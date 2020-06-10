# GradCam to better understand your network
This git is a user friendly implementation of GradCam to better understand the behaviour of any CNN-based computer vision network.
Paper at: https://arxiv.org/pdf/1610.02391

![alt_text](https://upload-images.jianshu.io/upload_images/415974-0147c44dcfb8cc1c.jpg)


The input needed is the pre-trained network in question, and at least one image to be used for inference. The last layer is visualized since it contains highest level of information.
The output is a superposition of the image and a heatmap indicating where the CNN layer is looking.

There are two main ways to use this git, either apply GradCam on one a single layer on multiple images,
or apply GradCam on a single image for multiple layers. 

<ins>Command for a single layer on multiple images</ins>
`python main.py --model_path vgg16 --conv2d_backcount 1 --images_path data --labels_path imagenet1000_labels.txt --n_images 3 --show`

<ins>Command for multiple layers on a single image</ins>
`python main.py --model_path vgg16 --conv2d_backcount 1 3 4 --images_path data --labels_path imagenet1000_labels.txt --show`

1. **model_path**: path to torch saved model (using torch.save())
2. **conv2d_backcount**: positive integer or a list of positive integers; CNN layer to visualize, counting from behind. By default equal to 1.
3. **images_path**: path pointing to the root of the images folder (not to the directory containing the images, but one level before).
4. **save_dir**: directory to save images on which gradcam was applied.
5. **n_images**: number of images used for inference.
6. **imageNet_labels**: add --imageNet_labels if labels used are from ImageNet.
7. **show**: add --show in command to show plots of GradCam. By default, plots will not be shown.
8. **labels_path**: path to .txt file containing a dictionary of your labels in the following format:

    {0: 'cat',                        
     1: 'dog',                        
     2: 'person'}
 
 --labels_path not needed if --imageNet_labels flag is used.
 
 **All images are saved in --save_dir directory**
 
<ins>Example of figures for a single layer with multiple images:</ins>
We always show the top three predictions and the worst prediction.
![alt_text](multiple_layers/image0.jpg)
![alt_text](multiple_layers/image4.jpg)
![alt_text](multiple_layers/image7.jpg)
 
<ins>Example of figures for a single layer with multiple images:</ins>
We always show the top three predictions and the worst prediction.
![alt_text](multiple_layers/layer1.jpg)
![alt_text](multiple_layers/layer2.jpg)
![alt_text](multiple_layers/layer3.jpg)

**About preprocessing**
We used the default preprocessing from ImageNet, and it is defined in utils.py
If another preprocessing is needed, the function should be replaced inside utils.py
