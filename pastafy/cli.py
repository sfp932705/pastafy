"""Console script for pastafy."""
import argparse
import os
import sys
import time
from pathlib import Path
from scipy.optimize import fmin_l_bfgs_b
from .pastafy import *


def parse_args():
    """
    Arg parser.
    """

    gpu_msg = "Whether to use GPU or not. Defaults to False."
    img_msg = "Images to be used. First one should be the content image (you) and second one the style (kitty). " \
              "Alternatively, you could specify any desired content and style."
    ratio_msg = "Ratio of the weights assigned to the style and content image (Alpha Beta ratio). Defaults to 0.01."
    iter_msg = "Number of iterations. Defaults to 10."
    layer_msg = "Feature layer to be extracted. Defaults to 4."
    output_msg = "Name of output image. Defaults to a concatenation of style image name and content image name."

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', dest='gpu', help=gpu_msg, type=bool, default=False)
    parser.add_argument('-i', '--images', dest='img', nargs=2, help=img_msg, type=str, required=True)
    parser.add_argument('-n', '--number', dest='iters', help=iter_msg, type=int, default=10)
    parser.add_argument('-l', '--layer', dest='layer', help=layer_msg, type=int, default=4)
    parser.add_argument('-o', '--output', dest='output', help=output_msg, type=str, default=None)
    parser.add_argument('-r', '--ratio', dest='ratio', help=ratio_msg, type=float, default=0.01)

    args = parser.parse_args()
    if args.output == None: args.output = Path('images',f'{Path(args.img[0]).stem}_{Path(args.img[1]).name}')
    if not args.gpu: os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    return args


def main():
    """
    Console script for catify.

    CLI Args:
        -i/--images: Images to be used. First one should be the content image (you) and second one the style (kitty). Alternatively, you could specify any desired content and style.

        -n/--number: Number of iterations. Defaults to 10.

        -l/--layer: Feature layer to be extracted. Defaults to 3.

        -o/--output: Name of output image. Defaults to a concatenation of style image name and content image name.

        -r/--ratio: Ratio of the weights assigned to the style and content image (Alpha Beta ratio). Defaults to 0.1.
    """

    args = parse_args()
    disable_eager()
    height, width = img_shape(args.img[0])
    content, style = preprocess_image(args.img[0]), preprocess_image(args.img[1])
    combination, input_tensor = combine_content_style(content, style)
    layers, layer_name = get_VGG16_layers(input_tensor, args.layer)
    loss = generate_loss_from_layers(layers, layer_name, combination, args.ratio)
    grads = keras_gradients(loss, combination)
    outputs = [loss, grads]
    f_outputs = keras_function([combination], outputs)
    evaluator = Evaluator(f_outputs)
    x = preprocess_image(args.img[0])

    for i in range(args.iters):
        print(f'Start of iteration {i}')
        start_time = time.time()
        x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
        total_time = time.time() - start_time
        print(f'Iteration {i} completed in {total_time:.2f}s.')

    save_output(x.reshape(content.shape[1:]), (width, height), args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
