from torchvision import models
from GradCam import GradCam
import urllib3
import shutil

# First we download the ImageNet Labels, used for better analysis of the results
http = urllib3.PoolManager()
url ="https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
idx2label_path = "imagenet1000_labels.txt"
# r = http.request('GET', link)
with http.request('GET', url, preload_content=False) as r, open(idx2label_path, 'wb') as out_file:
    shutil.copyfileobj(r, out_file)


data_path = "data"
model = models.vgg16(pretrained=True)
layer = model.features[28]
grad_cam = GradCam(model, layer, data_path, idx2label_path)
grad_cam.lunch_grad_cam(n_images=5)
grad_cam.plot_grad_cam()

