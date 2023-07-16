import os
import argparse

import torch
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from timm.data import create_dataset, create_loader, resolve_data_config
from timm.layers import apply_test_time_pool
from timm.models import create_model
from timm.utils import ParseKwargs


parser = argparse.ArgumentParser(description='PyTorch ImageNet TorchCAM')
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
parser.add_argument('--layer-names', nargs="*", default=['conv_head'], type=str, help='layer names')
parser.add_argument('--model-kwargs', nargs='*', default={}, action=ParseKwargs)


def main():
    args = parser.parse_args()
    os.makedirs("torchcam", exist_ok=True)

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
    print(model)
    data_config = resolve_data_config(vars(args), model=model)
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

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
    for batch_idx, (input, _) in enumerate(loader):
        cam_extractor = SmoothGradCAMpp(model, target_layer=args.layer_names)
        img = (input[0].cpu().numpy().transpose((1, 2, 0)) + 1.0) / 2.0 * 255.0
        img = img.astype(np.uint8)
        out = model(input)
        cams = cam_extractor(out.squeeze(0).argmax().item(), out)
        for name, cam in zip(cam_extractor.target_names, cams):
            result = overlay_mask(Image.fromarray(img), to_pil_image(cam.squeeze(0), mode='F'), alpha=0.5)
            fig, ax = plt.subplots()
            im = ax.imshow(result, cmap="jet")
            ax.axis('off')
            ax.set_title(name)
            fig.colorbar(im)
            plt.savefig(f"torchcam/{filenames[batch_idx]}_{name}.png")


if __name__ == '__main__':
    main()
