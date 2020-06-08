import time, os
from torch.nn import functional as F
import matplotlib.pyplot as plt
from copy import deepcopy
from Hook import Hook
from utils import *


class GradCam:
    def __init__(self, model, data_path, idx2label_path, save_dir=None, show=True):
        if save_dir is None and show is not True:
            raise ValueError("Can't have show=False and save_dir=None")
        self.model = model
        self.dataset = preprocess_image(data_path)
        self.save_dir = save_dir
        self.show = show
        self.hook = Hook(self.model)
        self.n_classes = self.get_n_classes() # to fix, ret
        # rieve from model design
        with open(idx2label_path) as f:
            self.idx2label = eval(f.read())

    def get_n_classes(self):
        size=50
        done=False
        while not done:
            try:
                n_classes = int(np.prod(self.model(torch.randn(1,3,size,size)).size()))
                done=True
            except:
                size += 50
            if size>1000:
                done=True
                raise RuntimeError('Cant get number of classes of model.')
        return n_classes

    def lunch_grad_cam(self, n_images=20, random_seed=42):
        start = time.time()
        np.random.seed(random_seed)
        idx_images = np.random.randint(low=0, high=len(self.dataset), size=n_images)
        attentions = {i.item():[] for i in idx_images}

        for index in idx_images:
            image, _ = self.dataset[index]
            image = image.view(1, 3, 224, 224)
            output = self.model(image)
            indices = torch.cat((torch.topk(output, 3)[1], torch.topk(-output, 1)[1]), 1)[0]
            for prediction in indices:
                # setting out gradient to 1 on the wanted class and 0 elsewhere
                my_gradients =  torch.zeros(self.n_classes)
                my_gradients[prediction] = 1
                my_gradients.unsqueeze_(0)

                # back prpoagation
                loss = torch.sum(output * my_gradients)

                self.model.zero_grad()
                loss.backward(retain_graph=True)

                # Constructing heat map
                forward_weights = self.hook.forward
                grad_output_mean = self.hook.backward
                layer_shape = forward_weights.shape[-2:]

                heat_map = np.zeros(layer_shape)
                for i, alpha in enumerate(grad_output_mean):
                    heat_map += alpha * forward_weights[0, i, : , :]
                # np relu
                heat_map = np.maximum(heat_map, 0)

                # upsampling
                heat_map = torch.Tensor(heat_map).unsqueeze(0)
                heat_map.unsqueeze_(0)
                heat_map = F.interpolate(heat_map, scale_factor=16, mode='bilinear', align_corners=False)

                # normalizing
                heat_map.squeeze_()
                heat_map = heat_map / torch.max(heat_map)
                image_copy = deepcopy(image.squeeze(0).permute(1, 2, 0))

                image_heated = heat_image(denormalize(image_copy), heat_map)
                del image_copy
                attentions[index] += [(image_heated, prediction)]
        end = time.time()
        self.attentions = attentions

    def plot_grad_cam(self):
        print('top 3 labels are very close (indian elephant, african elephant..). We also show the lowest score label')
        if not hasattr(self, 'attentions'):
            raise RuntimeError('run lunch_grad_cam first')

        for idx in self.attentions:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15,5))

            image, label = self.attentions[idx][0]
            label = label.item()
            label = self.idx2label[label].split(',')[0]
            ax1.imshow(image)
            ax1.set_title(f"{label}")

            image, label = self.attentions[idx][1]
            label = label.item()
            label = self.idx2label[label].split(',')[0]
            ax2.imshow(image)
            ax2.set_title(f"{label}")

            image, label = self.attentions[idx][2]
            label = label.item()
            label = self.idx2label[label].split(',')[0]
            ax3.imshow(image)
            ax3.set_title(f"{label}")

            image, label = self.attentions[idx][3]
            label = label.item()
            label = self.idx2label[label].split(',')[0]
            ax4.imshow(image)
            ax4.set_title(f"lowest score: {label}")

            if self.save_dir is not None:
                if not os.path.isdir(self.save_dir):
                    os.mkdir(self.save_dir)
                image_loc = self.save_dir+'/image'+str(idx)+'.jpg'
                plt.savefig(image_loc)
            if self.show:
                plt.show()

