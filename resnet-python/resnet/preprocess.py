"""preprocess"""
import numpy as np
import cv2


def decode_image(im_file):
    """read rgb image
    Args:
        im_file (str|np.ndarray): input can be image path or np.ndarray
        im_info (dict): info of image
    Returns:
        im (np.ndarray):  processed image (np.ndarray)
        im_info (dict): info of processed image
    """
    if isinstance(im_file, str):
        with open(im_file, "rb") as f:
            im_read = f.read()
        data = np.frombuffer(im_read, dtype="uint8")
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    else:
        im = im_file
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


class Resize(object):
    """resize image by target_size and max_size
    Args:
        target_size (int): the target size of image
        keep_ratio (bool): whether keep_ratio or not, default true
        interp (int): method of resize
    """

    def __init__(self, size=None, resize_short=None, interp=cv2.INTER_LINEAR):
        """
        重新调整图像的大小
        
        Args:
            size (int or list): 输出图像的大小，如果为整数则表示图像高度和宽度都等于size。如果为列表，则需要包含两个元素，分别表示高度和宽度。默认为None。
            resize_short (int): 将输入图像缩放到最短边至少为resize_short的长度。如果为None或小于等于0，则不执行缩放操作。默认为None。
            interp (int): OpenCV的插值方法，默认为cv2.INTER_LINEAR。
        
        Raises:
            ValueError: 如果size和resize_short都为空，则抛出此错误。
        
        """
        self.size = size
        self.interp = interp
        if resize_short is not None and resize_short > 0:
            self.resize_short = resize_short
            self.w = None
            self.h = None
        elif size is not None:
            self.resize_short = None
            self.w = size if type(size) is int else size[0]
            self.h = size if type(size) is int else size[1]
        else:
            raise ValueError(
                "invalid params for ReisizeImage for '\
                'both 'size' and 'resize_short' are None"
            )

    def __call__(self, im):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        img_h, img_w = im.shape[:2]
        if self.resize_short is not None:
            percent = float(self.resize_short) / min(img_w, img_h)
            w = int(round(img_w * percent))
            h = int(round(img_h * percent))
        else:
            w = self.w
            h = self.h
        im = cv2.resize(im, (w, h), interpolation=self.interp)
        return im


class Crop(object):
    """crop image"""

    def __init__(self, size):
        """初始化ResizeTransform对象
        
        Args:
            size: 需要调整大小的图像的尺寸。如果size是整型，则表示图像的高度和宽度都等于size；
                如果size是一个tuple或list，则表示图像的高度为该tuple中的第一个元素，宽度为该tuple中的第二个元素。
        
        Returns:
            None
        
        Raises:
            TypeError: 如果输入参数size不是整型或tuple或list类型。
        """
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size  # (h, w)

    def __call__(self, img):
        """
        将输入的图像缩放并居中，使得宽和高在指定的范围内。
        
        Args:
            img (ndarray): 需要调整大小的灰度图像。
            
        Returns:
            ndarray: 调整后的灰度图像，形状为(height, width)。
            
        """
        w, h = self.size
        img_h, img_w = img.shape[:2]
        w_start = (img_w - w) // 2
        h_start = (img_h - h) // 2

        w_end = w_start + w
        h_end = h_start + h
        img = img[h_start:h_end, w_start:w_end, :]
        return img


class NormalizeImage(object):
    """normalize image
    Args:
        mean (list): im - mean
        std (list): im / std
        is_scale (bool): whether need im / 255
        norm_type (str): type in ['mean_std', 'none']
    """

    def __init__(self, mean, std, is_scale=True, norm_type="mean_std"):
        """
        Args:
            mean (float): 均值
            std (float): 分布标准差
            is_scale (bool): 是否进行缩放
            norm_type (str): 归一化类型，支持 "mean_std" 和 "batch_minmax"
        
        """
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.norm_type = norm_type

    def __call__(self, im):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        im = im.astype(np.float32, copy=False)
        if self.is_scale:
            scale = 1.0 / 255.0
            im *= scale

        if self.norm_type == "mean_std":
            multiply_factor = 1 / np.array(self.std)
            subtract_factor = np.array(self.mean) / np.array(self.std)
            im = np.multiply(im, multiply_factor)
            im = np.subtract(im, subtract_factor)
        return im


class Permute(object):
    """permute image
    Args:
        to_bgr (bool): whether convert RGB to BGR
        channel_first (bool): whether convert HWC to CHW
    """

    def __init__(
        self,
    ):
        """
        重置初始化函数，用于重新实例化对象。
        
        Args:
            无参数。
        
        Returns:
            无返回值。
        
        """
        super(Permute, self).__init__()

    def __call__(self, im):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        im = im.transpose((2, 0, 1)).copy()
        return im


class Preprocess(object):
    """Preprocess interface"""

    def __init__(self, target_size=224):
        """
        从给定图像的短边和长边中选择目标大小，并将其缩放为固定大小。
        
        Args:
            resize_short (int): 将图像缩放到这个短边大小。
        
        Attributes:
            resizer (Resize): 用于缩放图像的对象。
            crop (Crop): 用于裁剪图像的对象。
            normalizer (NormalizeImage): 对图像进行归一化处理的对象。
            permutor (Permute): 对图像通道顺序进行重新排列的对象。
        
        """
        self.resizer = Resize(resize_short=256)
        self.crop = Crop(target_size)
        self.normalizer = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], is_scale=True
        )
        self.permutor = Permute()

    def __call__(self, image):
        """
        根据给定的图像文件路径，返回处理后的图像。
        
        Args:
            image (str): 图像文件的路径。
        
        Returns:
            numpy.ndarray: 返回处理后的图像，形状为[1, H, W, C]。其中H为高，W为宽，C为通道数。
        
        """
        image = decode_image(image)
        image = self.resizer(image)
        image = self.crop(image)
        image = self.normalizer(image)
        image = self.permutor(image)
        image = image.astype("float32")
        return np.expand_dims(image, 0)
