from GradCam import GradCam
import argparse
import urllib3
import shutil
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GradCam model')

    parser.add_argument('--model_path', type=str, required=True,
                        help='path for torch model, saved using torch.save()')
    parser.add_argument('--images_path', type=str, required=True,
                        help='path for images used for inference')
    parser.add_argument('--labels_path', type=str, required=False,
                        help='path to load index to label mapping, from .txt file')
    parser.add_argument('--imageNet_labels', default=False, action='store_true',
                        help='True if labels used are from ImageNet')
    parser.add_argument('--save_dir', type=str, required=False, default=None,
                        help='directory to save images on which gradcam was applied')
    parser.add_argument('--n_images', type=int, default=1,
                        help='number of images used for inference')
    parser.add_argument('--show', default=False, action='store_true',
                        help='set True to show plots. Default is True')

    args = parser.parse_args()
    print(args)
    if args.imageNet_labels:
        http = urllib3.PoolManager()
        url ="https://gist.githubusercontent.com/yrevar/942d3a0ac09ec9e5eb3a/raw/238f720ff059c1f82f368259d1ca4ffa5dd8f9f5/imagenet1000_clsidx_to_labels.txt"
        idx2label_path = "imagenet1000_labels.txt"
        with http.request('GET', url, preload_content=False) as r, open(idx2label_path, 'wb') as out_file:
            shutil.copyfileobj(r, out_file)
    else:
        idx2label_path = args.labels_path

    model = torch.load(args.model_path)
    grad_cam = GradCam(model, args.images_path, idx2label_path, save_dir=args.save_dir, show=args.show)
    grad_cam.lunch_grad_cam(n_images=args.n_images)
    grad_cam.plot_grad_cam()
