import numpy as np
from torchio import transforms as transforms
from torchio import Subject

class RandomResizedCrop(transforms.augmentation.RandomTransform, transforms.SpatialTransform):
    def __init__(self, output_size=256, **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size
    
    def apply_transform(self, subject: Subject) -> Subject:
        for image in self.get_images(subject):
            self._random_crop_image(image, self.output_size)
        return subject
    
    @staticmethod
    def _random_crop_image(image, output_size):
        data = image.data
        d_x, d_y, d_z = data.shape
        x, y, z = output_size, output_size, output_size
        p_x, p_y, p_z = x/d_x, y/d_y, z/d_z
        r_x, r_y, r_z = np.random.random(3)
        
        x_0 = int(d_x * p_x * r_x)
        y_0 = int(d_y * p_y * r_y)
        z_0 = int(d_z * p_z * r_z)

        x_1 = x_0 + x
        y_1 = y_0 + y
        z_1 = z_0 + z

        data_crop = data[x_0:x_1, y_0:y_1, z_0:z_1]
        image.set_data(data_crop)

class CenterCrop(transforms.augmentation.RandomTransform, transforms.SpatialTransform):
    def __init__(self, output_size=256, **kwargs):
        super().__init__(**kwargs)
        self.output_size = output_size
    
    def apply_transform(self, subject: Subject) -> Subject:
        for image in self.get_images(subject):
            self._center_crop(image, self.output_size)
        return subject
    
    @staticmethod
    def _center_crop(image, output_size):
        data = image.data
        d_x, d_y, d_z = data.shape
        x, y, z = output_size, output_size, output_size
        
        x_0 = (d_x - x) // 2
        y_0 = (d_y - y) // 2
        z_0 = (d_z - z) // 2

        x_1 = (d_x + x) // 2
        y_1 = (d_y + y) // 2
        z_1 = (d_z + z) // 2

        data_crop = data[x_0:x_1, y_0:y_1, z_0:z_1]
        image.set_data(data_crop)        

class NLSTTrainDataTransform(object):
    """
    Transforms for SimCLR

    Transform::

        RandomResizedCrop(size=self.input_height)
        RandomHorizontalFlip()
        RandomApply([color_jitter], p=0.8)
        RandomGrayscale(p=0.2)
        GaussianBlur(kernel_size=int(0.1 * self.input_height))
        transforms.ToTensor()

    Example::

        from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform

        transform = SimCLRTrainDataTransform(input_height=32)
        x = sample()
        (xi, xj) = transform(x)
    """

    def __init__(
        self, input_height: int = 224, gaussian_blur: bool = True, jitter_strength: float = 1., normalize=None
    ) -> None:


        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.normalize = normalize

        data_transforms = [
            RandomResizedCrop(self.input_height),
            transforms.RandomFlip(axes=('lr', 'ap'), flip_probability=0.5)
        ]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1

            data_transforms.append(GaussianBlur(kernel_size=kernel_size, p=0.5))

        self.train_transform = transforms.Compose(data_transforms)

        # add online train transform of the size of global view
        self.online_transform = transforms.Compose([
            RandomResizedCrop(self.input_height),
            transforms.RandomFlip(axes=('lr', 'ap'), flip_probability=0.5)
        ])

    def __call__(self, sample):
        transform = self.train_transform

        xi = transform(sample)
        xj = transform(sample)

        return xi.data, xj.data, self.online_transform(sample).data


class NLSTEvalDataTransform(NLSTTrainDataTransform):
    """
    Transforms for SimCLR

    Transform::

        Resize(input_height + 10, interpolation=3)
        transforms.CenterCrop(input_height),
        transforms.ToTensor()

    Example::

        from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform

        transform = SimCLREvalDataTransform(input_height=32)
        x = sample()
        (xi, xj) = transform(x)
    """

    def __init__(
        self, input_height: int = 224, gaussian_blur: bool = True, jitter_strength: float = 1., normalize=None
    ):
        super().__init__(
            normalize=normalize,
            input_height=input_height,
            gaussian_blur=gaussian_blur,
            jitter_strength=jitter_strength
        )

        # replace online transform with eval time transform
        self.online_transform = transforms.Compose([
            transforms.Resize(int(self.input_height + 0.1 * self.input_height)),
            CenterCrop(self.input_height),
        ])


class NLSTFinetuneTransform(object):

    def __init__(
        self,
        input_height: int = 224,
        jitter_strength: float = 1.,
        normalize=None,
        eval_transform: bool = False
    ) -> None:

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.normalize = normalize

        if not eval_transform:
            data_transforms = [
                RandomResizedCrop(size=self.input_height),
                transforms.RandomFlip(axes=('lr', 'ap'), flip_probability=0.5)
            ]
        else:
            data_transforms = [
                transforms.Resize(int(self.input_height + 0.1 * self.input_height)),
                CenterCrop(self.input_height)
            ]

        if normalize is None:
            final_transform = transforms.ToTensor()
        else:
            final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        data_transforms.append(final_transform)
        self.transform = transforms.Compose(data_transforms)

    def __call__(self, sample):
        return self.transform(sample)

class GaussianBlur(object):
    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):

        self.min = min
        self.max = max

        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.p:
            blur = transforms.RandomBlur(std=(self.min, self.max))

        return blur(sample)