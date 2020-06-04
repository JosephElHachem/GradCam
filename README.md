# GradCam
An easy to use implementation of GradCam to better understand computer vision networks' behaviour.

Example to run:

python main.py --model_path vgg16 --images_path data --labels_path imagenet1000_labels.txt --n_images 3 --show

model_path: path to torch saved model (using torch.save())

images_path: path pointing to the root of the images folder (not to the directory containing the images, but one level before)

save_dir: directory to save images on which gradcam was applied

n_images: number of images used for inference

imageNet_labels: True if labels used are from ImageNet

labels_path: path to .txt file containing a dictonary of your labels in the following format

labels.txt

----------------------------
{0: 'cat',
 1: 'dog',
 2: 'person'}
 ----------------------------
 
 show: add --show in command to show plots of GradCam. By default, plots will not be shown
