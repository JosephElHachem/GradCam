import os
from torch.nn import functional as F
import matplotlib.pyplot as plt
from copy import deepcopy
from Hook import Hook
from utils import *
from numpy.random import default_rng


class GradCam:
    def __init__(self, model, data_path, idx2label_path, conv2d_backcount=1, multiple_layers=[1], save_dir=None, show=True):
        if save_dir is None and show is not True:
            raise ValueError("Can't have show=False and save_dir=None")
        self.model = model
        self.dataset = preprocess_image(data_path)
        self.save_dir = save_dir
        self.show = show
        self.conv2d_backcount = conv2d_backcount
        self.multiple_layers = multiple_layers
        self.hook = None
        self.n_classes = self.get_n_classes() # to fix, ret
        self.save_name = None # will be set to 'image' if using launch_images, and to 'layer' if using launch_layers
        # rieve from model design
        with open(idx2label_path) as f:
            self.idx2label = eval(f.read())
        self.attentions = None

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

    def grad_cam_image(self, image, cnn_layer_index):
        '''
        :param image: tensor extracted from self.dataset
        :param cnn_layer_index: int; index of cnn layer to visualize, counting from behind
        :return:
         applies gradcam and returns list of tuples (image_heated, prediction)
         for the top 3 predictions and the worst prediction
        '''
        # first hook
        self.hook = Hook(self.model, conv2d_backcount=cnn_layer_index)

        results = []
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

            # upsampling scale factor
            scale_factor = image.shape[-2] / layer_shape[0]

            # upsampling
            heat_map = torch.from_numpy(heat_map).unsqueeze(0)
            heat_map.unsqueeze_(0)
            heat_map = F.interpolate(heat_map, scale_factor=scale_factor, mode='bilinear', align_corners=False) # SCALE FACTOR

            # normalizing
            heat_map.squeeze_()
            heat_map = heat_map / torch.max(heat_map)
            image_copy = deepcopy(image.squeeze(0).permute(1, 2, 0))

            image_heated = heat_image(denormalize(image_copy), heat_map)
            del image_copy

            results += [(image_heated, prediction)]
        return results

    def launch_layers(self, image=None, random_seed=42):
        '''
        :param image: image to plot
        :param random_seed: random seed when choosing index of image randomly
        :return: None, but builds self.attentions
        '''
        np.random.seed(random_seed)
        self.save_name = 'layer'
        self.attentions = {l_index:[] for l_index in self.multiple_layers}
        if image is None:
            image = self.dataset[np.random.randint(0, len(self.dataset))][0]
        for l_index in self.attentions.keys():
            print(f'layer {l_index} done')
            self.attentions[l_index] = self.grad_cam_image(image, l_index)
        pass

    def launch_images(self, n_images=20, random_seed=42):
        '''
        :param n_images: number of images to apply self.grad_cam_image
        :param random_seed: random seed to choose images
        :return: None; builds self.attentions
        '''
        np.random.seed(random_seed)
        self.save_name = 'image'
        rng = default_rng()
        idx_images = rng.choice(len(self.dataset), size=n_images, replace=False)
        self.attentions = {i.item():[] for i in idx_images}
        for index in idx_images:
            self.attentions[index] += self.grad_cam_image(self.dataset[index][0], self.conv2d_backcount)
            print(f'image {index} done')

    def plot_grad_cam(self):
        if not hasattr(self, 'attentions'):
            raise RuntimeError('run launch_images first')

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
                image_loc = self.save_dir+'/'+self.save_name+str(idx)+'.jpg'
                plt.savefig(image_loc)
            if self.show:
                plt.show()
