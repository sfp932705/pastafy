"""Main module."""
import numpy as np
from PIL import Image
from tensorflow.compat.v1 import disable_eager_execution
from tensorflow.keras import backend
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.framework.ops import Tensor
from tensorflow.python.ops.resource_variable_ops import ResourceVariable

vgg_layer_mapping = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2',
                     'block3_conv3', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block5_conv1', 'block5_conv2',
                     'block5_conv3']


class Evaluator(object):
    def __init__(self, func):
        self.func = func
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self.eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

    def eval_loss_and_grads(self, x: np.ndarray, height: int = 512, width: int = 512):
        x = x.reshape((1, height, width, 3))
        loss_value, grads = self.func(x)
        grad_values = grads[0].flatten().astype('float64')
        return loss_value, grad_values


def disable_eager():
    """
    Disable eager mode to keep tensorflow v1 compatibility.
    """

    disable_eager_execution()


def img_shape(img_path: str) -> tuple:
    """
    Return the shape of an image array.

    Args:
        img_path: Path to image.

    Returns:
        Height and width of the image.
    """

    img = load_img(img_path)
    img = img_to_array(img)

    return (img.shape)[:2]


def preprocess_image(img_path: str, height: int = 512, width: int = 512) -> np.ndarray:
    """
    Loads and adequate image to the format the VGG16 requires.

    Args:
        img_path: Path to image.

        height: Height of target size. Depends on the model used (512 for VGG16).

        width: Width of target size. Depends on the model used (512 for VGG16).

    Returns:
        Loaded and preprocessed image.
    """

    img = load_img(img_path, target_size=(height, width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    return img


def combine_content_style(content: np.ndarray, style: np.ndarray) -> tuple:
    """
    Combines style and content image and generates input tensor using this combination.

    Args:
        content: Array containing the content image.

        style: Array containing the style image.

    Returns:
        Tuple of Tensorflow Tensors: combination tensor and input tensor.
    """

    content, style = backend.variable(content), backend.variable(style)
    combination = backend.placeholder(content.shape)
    input_tensor = backend.concatenate([content, style, combination], axis=0)

    return combination, input_tensor


def get_VGG16_layers(input_tensor: Tensor, layer: int) -> tuple:
    """
    Get VGG16 layers with ImageNet weights and extract specified feature layer.

    Args:
        input_tensor: Input tensor for VGG16.

        layer: Feature layer to be extracted.

    Returns:
        VGG16 layers and desired feature layer name to be extracted.
    """

    model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
    layers = {_layer.name: _layer.output for _layer in model.layers}

    return layers, vgg_layer_mapping[layer]
    # return layers, layers[vgg_layer_mapping[layer]]


def content_loss(content: Tensor, combination: Tensor) -> Tensor:
    """
    Computes content loss based on content and combination.

    Args:
        content: Content tensor.

        combination: Combination tensor.

    Returns:
        Loss tensor.
    """

    return backend.sum(backend.square(content - combination))


def style_loss(style: Tensor, combination: Tensor) -> Tensor:
    """
    Computes style loss based on style and combination.

    Args:
        style: Content tensor. Must be 4D (batch,.,.,.) or 3D.

        combination: Combination tensor. Must be 4D (batch,.,.,.) or 3D.

    Returns:
        Loss tensor.
    """

    indeces = (1, 2, 3) if len(style.shape) == 4 else (0, 1, 2)
    S, C = gram_matrix(style), gram_matrix(combination)
    size, channels = style.shape[indeces[0]] * style.shape[indeces[1]], style.shape[indeces[2]]

    return backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def gram_matrix(x: Tensor) -> Tensor:
    """
    Computes the Gram Matrix.

    Args:
        Tensor to be considered in Gramian Matrix calculation. Must be 4D (batch,.,.,.) or 3D.

    Returns:
        Gram Matrix.
    """

    order = (0, 3, 1, 2) if len(x.shape) == 4 else (2, 0, 1)
    features = backend.batch_flatten(backend.permute_dimensions(x, order))

    return backend.dot(features, backend.transpose(features))


def total_variation_loss(x: Tensor) -> Tensor:
    """
    Computes total loss.

    Args:
        x: Tensor considered. Must be 4D (batch,.,.,.) or 3D.
    Returns:
        Computed loss.

    """

    height, width = (x.shape[1], x.shape[2]) if len(x.shape) == 4 else (x.shape[0], x.shape[1])
    a = backend.square(x[:, :height - 1, :width - 1, :] - x[:, 1:, :width - 1, :])
    b = backend.square(x[:, :height - 1, :width - 1, :] - x[:, :height - 1, 1:, :])

    return backend.sum(backend.pow(a + b, 1.25))


def keras_variable(value: float) -> ResourceVariable:
    """

    Args:
        value: Float/int value to be stored as a Keras variable.

    Returns:
        Keras varibale.
    """

    return backend.variable(value)


def keras_function(inputs: list, outputs: list) -> list:
    """
    Instantiates a Keras function as described in tensorflow docs.

    Args:
        inputs: List of placeholder tensors.

        outputs: List of output tensors.

    Returns:
        Output values as Numpy arrays.
    """

    return backend.function(inputs, outputs)


def keras_gradients(loss: ResourceVariable, variables: list) -> list:
    """
    Returns the gradients of loss w.r.t. variables as described in tensorflow docs.

    Args:
        loss: Scalar tensor to minimize.

        variables: List of variables.

    Returns:
        A gradients tensor.
    """

    return backend.gradients(loss, variables)


def generate_loss_from_layers(model_layers: dict, desired_layer: Tensor, combination: Tensor, ratio: float,
                              content_weight: float = 0.025, total_variation_weight: float = 1) -> Tensor:
    """
    Generates loss tensor.

    Args:
        model_layers: Dict of model layers' names and outputs.

        desired_layer: Desired layer name.

        ratio: Ratio of the weights assigned to the style and content image (Alpha Beta ratio).

    Returns:
        Loss tensor.
    """

    feature_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3']
    style_weight = content_weight / ratio
    loss = keras_variable(0.)
    combination_features = model_layers[desired_layer][2, :, :, :]
    loss = loss + content_weight * content_loss(model_layers[desired_layer][0, :, :, :], combination_features)
    for layer_name in feature_layers:
        layer_features = model_layers[layer_name]
        style_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_features, combination_features)
        loss = loss + (style_weight / len(feature_layers)) * sl

    loss = loss + total_variation_weight * total_variation_loss(combination)

    return loss


def save_output(y: np.ndarray, size: tuple, output: str):
    """
    Prepares and saves ouput image.

    Args:
        y: Array with results that must be prepared and converted to image.

        size: (width,height) resolution for output image.

        output: Output image file name.
    """

    y[:, :, 0] += 103.939
    y[:, :, 1] += 116.779
    y[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    y = y[:, :, ::-1]
    y = np.clip(y, 0, 255).astype('uint8')
    result = Image.fromarray(y).resize(size, Image.LANCZOS)
    result.save(output)
