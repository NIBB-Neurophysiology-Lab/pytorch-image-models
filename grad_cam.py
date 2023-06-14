import os
import argparse

import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import matplotlib.pyplot as plt

from timm.data import create_dataset, create_loader, resolve_data_config
from timm.layers import apply_test_time_pool
from timm.models import create_model
from timm.utils import ParseKwargs


parser = argparse.ArgumentParser(description='PyTorch ImageNet GradCAM')
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (*deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--model', '-m', metavar='MODEL', default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--in-chans', type=int, default=None, metavar='N',
                    help='Image input channels (default: None => 3)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--layer-name', default='conv_head', type=str, help='layer name')
parser.add_argument('--model-kwargs', nargs='*', default={}, action=ParseKwargs)


def main():
    args = parser.parse_args()
    os.makedirs("gradcam", exist_ok=True)

    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]
    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=in_chans,
        pretrained=args.pretrained,
        checkpoint_path=args.checkpoint,
        **args.model_kwargs,
    )
    data_config = resolve_data_config(vars(args), model=model)
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config)

    target_layers = [getattr(model, args.layer_name)]
    cam = GradCAM(model = model, target_layers = target_layers, use_cuda = torch.cuda.is_available())

    root_dir = args.data or args.data_dir
    dataset = create_dataset(
        root=root_dir,
        name=args.dataset,
        split=args.split,
    )

    if test_time_pool:
        data_config['crop_pct'] = 1.0

    workers = 1 if 'tfds' in args.dataset or 'wds' in args.dataset else args.workers
    loader = create_loader(
        dataset,
        batch_size=1,
        use_prefetcher=True,
        num_workers=workers,
        **data_config,
    )

    filenames = loader.dataset.filenames()
    for batch_idx, (input, target) in enumerate(loader):
        vis_image = (input[0].cpu().numpy().transpose((1, 2, 0)) + 1.0) / 2.0
        label = [ClassifierOutputTarget(target[0])]
        grayscale_cam = cam(input_tensor=input, targets=label)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(vis_image, grayscale_cam, use_rgb=True)
        fig, ax = plt.subplots()
        im = ax.imshow(visualization, cmap="jet")
        fig.colorbar(im)
        plt.savefig(f"gradcam/{filenames[batch_idx]}.png")


if __name__ == '__main__':
    main()
