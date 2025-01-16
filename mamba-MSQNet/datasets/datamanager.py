import itertools

from .transforms_ss import *
from torchvision.transforms import Compose
from .datasets import AnimalKingdom, MammalNet, BaboonLandDataset
from torch.utils.data import Dataset, DataLoader, RandomSampler, DistributedSampler
import numpy as np
import torch


class LimitDataset(Dataset):
    """
    To ensure a constant number of samples are retrieved from the dataset we use this
    LimitDataset wrapper. This is necessary because several of the underlying videos
    may be corrupted while fetching or decoding, however, we always want the same
    number of steps per epoch.
    """

    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )

    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos


class DataManager():
    def __init__(self, args, path):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.path = path
        self.dataset = args.dataset
        self.total_length = args.total_length
        self.test_part = args.test_part
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.distributed = args.distributed

    def _check(self, ):
        datasets_list = ["animalkingdom", "baboonland", 'mammalnet']
        if (self.dataset not in datasets_list):
            raise Exception("[ERROR] The dataset " + str(self.dataset) + " is not supported!")

    def get_num_classes(self, ):
        self._check()
        if (self.dataset == "animalkingdom"):
            return 140
        elif self.dataset == "baboonland":
            return 13
        elif (self.dataset == "mammalnet"):
            return 12

    def get_act_dict(self, ):
        self._check()
        animalkingdom_dict = {'Abseiling': 0, 'Attacking': 1, 'Attending': 2, 'Barking': 3, 'Being carried': 4,
                              'Being carried in mouth': 5, 'Being dragged': 6, 'Being eaten': 7, 'Biting': 8,
                              'Building nest': 9, 'Calling': 10, 'Camouflaging': 11, 'Carrying': 12,
                              'Carrying in mouth': 13, 'Chasing': 14, 'Chirping': 15, 'Climbing': 16, 'Coiling': 17,
                              'Competing for dominance': 18, 'Dancing': 19, 'Dancing on water': 20, 'Dead': 21,
                              'Defecating': 22, 'Defensive rearing': 23, 'Detaching as a parasite': 24, 'Digging': 25,
                              'Displaying defensive pose': 26, 'Disturbing another animal': 27, 'Diving': 28,
                              'Doing a back kick': 29, 'Doing a backward tilt': 30, 'Doing a chin dip': 31,
                              'Doing a face dip': 32, 'Doing a neck raise': 33, 'Doing a side tilt': 34,
                              'Doing push up': 35, 'Doing somersault': 36, 'Drifting': 37, 'Drinking': 38, 'Dying': 39,
                              'Eating': 40, 'Entering its nest': 41, 'Escaping': 42, 'Exiting cocoon': 43,
                              'Exiting nest': 44, 'Exploring': 45, 'Falling': 46, 'Fighting': 47, 'Flapping': 48,
                              'Flapping tail': 49, 'Flapping its ears': 50, 'Fleeing': 51, 'Flying': 52,
                              'Gasping for air': 53, 'Getting bullied': 54, 'Giving birth': 55, 'Giving off light': 56,
                              'Gliding': 57, 'Grooming': 58, 'Hanging': 59, 'Hatching': 60,
                              'Having a flehmen response': 61, 'Hissing': 62, 'Holding hands': 63, 'Hopping': 64,
                              'Hugging': 65, 'Immobilized': 66, 'Jumping': 67, 'Keeping still': 68, 'Landing': 69,
                              'Lying down': 70, 'Laying eggs': 71, 'Leaning': 72, 'Licking': 73,
                              'Lying on its side': 74, 'Lying on top': 75, 'Manipulating object': 76, 'Molting': 77,
                              'Moving': 78, 'Panting': 79, 'Pecking': 80, 'Performing sexual display': 81,
                              'Performing allo-grooming': 82, 'Performing allo-preening': 83,
                              'Performing copulatory mounting': 84, 'Performing sexual exploration': 85,
                              'Performing sexual pursuit': 86, 'Playing': 87, 'Playing dead': 88, 'Pounding': 89,
                              'Preening': 90, 'Preying': 91, 'Puffing its throat': 92, 'Pulling': 93, 'Rattling': 94,
                              'Resting': 95, 'Retaliating': 96, 'Retreating': 97, 'Rolling': 98, 'Rubbing its head': 99,
                              'Running': 100, 'Running on water': 101, 'Sensing': 102, 'Shaking': 103,
                              'Shaking head': 104, 'Sharing food': 105, 'Showing affection': 106, 'Sinking': 107,
                              'Sitting': 108, 'Sleeping': 109, 'Sleeping in its nest': 110, 'Spitting': 111,
                              'Spitting venom': 112, 'Spreading': 113, 'Spreading wings': 114, 'Squatting': 115,
                              'Standing': 116, 'Standing in alert': 117, 'Startled': 118, 'Stinging': 119,
                              'Struggling': 120, 'Surfacing': 121, 'Swaying': 122, 'Swimming': 123,
                              'Swimming in circles': 124, 'Swinging': 125, 'Tail swishing': 126, 'Trapped': 127,
                              'Turning around': 128, 'Undergoing chrysalis': 129, 'Unmounting': 130, 'Unrolling': 131,
                              'Urinating': 132, 'Walking': 133, 'Walking on water': 134, 'Washing': 135, 'Waving': 136,
                              'Wrapping itself around prey': 137, 'Wrapping prey': 138, 'Yawning': 139}
        baboonland_dict = {"Walking/Running": 0, "Sitting/Standing": 1, "Fighting/Playing": 2, "Self-Grooming": 3,
                           "Being Groomed": 4, "Grooming Somebody": 5, "Mutual Grooming": 6, "Infant-Carrying": 7,
                           "Foraging": 8, "Drinking": 9, "Mounting": 10, "Sleeping": 11, "Occluded": 12}
        mammalnet_dict = {"drinks water": 0, "sleeps": 1, "eats food": 2, "mates with other animals": 3,
                          "nurses or breastfeeds its baby": 4, "pees": 5, "grooms/cleans itself or other animal": 6,
                          "poops": 7, "fights against other animals": 8, "hunts other animals": 9, "vomits": 10,
                          "gives birth to a baby": 11}
        if self.dataset == "animalkingdom":
            return animalkingdom_dict
        elif self.dataset == "baboonland":
            return baboonland_dict
        elif (self.dataset == "mammalnet"):
            return mammalnet_dict


    def get_train_transforms(self, ):
        """Returns the training torchvision transformations for each dataset/method.
           If a new method or dataset is added, this file should by modified
           accordingly.
        Args:
          method: The name of the method.
        Returns:
          train_transform: An object of type torchvision.transforms.
        """
        self._check()
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        input_size = 224

        if self.dataset == 'ava':
            # video_transforms = create_video_transform(mode='train',
            #                                           video_key='video',
            #                                           remove_key=['video_name', 'video_index', 'clip_index', 'aug_index', 'boxes', 'extra_info'],
            #                                           num_samples=8,
            #                                           convert_to_float=False)
            # labels_transforms = ApplyTransformToKey(key='labels',
            #     transform=Lambda(lambda nest_list: np.array([(1) if i in set([item-1 for _list in nest_list for item in _list]) else (0) for i in range(80)])))
            # transforms = Compose([video_transforms, labels_transforms])
            raise NotImplementedError
        else:
            unique = Compose([GroupMultiScaleCrop(input_size, [1, .875, .75, .66]),
                              GroupRandomHorizontalFlip(True),
                              GroupRandomColorJitter(p=0.8, brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
                              GroupRandomGrayscale(p=0.2),
                              GroupGaussianBlur(p=0.0),
                              GroupSolarization(p=0.0)])
            common = Compose([Stack(roll=False),
                              ToTorchFormatTensor(div=True),
                              GroupNormalize(input_mean, input_std)])
            transforms = Compose([unique, common])
        return transforms

    def get_test_transforms(self, ):
        """Returns the evaluation torchvision transformations for each dataset/method.
           If a new method or dataset is added, this file should by modified
           accordingly.
        Args:
          method: The name of the method.
        Returns:
          test_transform: An object of type torchvision.transforms.
        """
        self._check()
        input_mean = [0.48145466, 0.4578275, 0.40821073]
        input_std = [0.26862954, 0.26130258, 0.27577711]
        input_size = 224
        scale_size = 256

        if self.dataset == 'ava':
            # video_transforms = create_video_transform(mode='val',
            #                                           video_key='video',
            #                                           remove_key=['video_name', 'video_index', 'clip_index', 'aug_index', 'boxes', 'extra_info'],
            #                                           num_samples=8,
            #                                           convert_to_float=False)
            # labels_transforms = ApplyTransformToKey(key='labels',
            #     transform=Lambda(lambda nest_list: np.array([(1) if i in set([item-1 for _list in nest_list for item in _list]) else (0) for i in range(80)])))
            # transforms = Compose([video_transforms, labels_transforms])
            raise NotImplementedError
        else:
            unique = Compose([GroupScale(scale_size),
                              GroupCenterCrop(input_size)])
            common = Compose([Stack(roll=False),
                              ToTorchFormatTensor(div=True),
                              GroupNormalize(input_mean, input_std)])
            transforms = Compose([unique, common])
        return transforms

    def get_train_loader(self, train_transform, drop_last=False):
        """Returns the training loader for each dataset.
           If a new method or dataset is added, this method should by modified
           accordingly.
        Args:
          path: disk location of the dataset.
          dataset: the name of the dataset.
          total_length: the number of frames in a video clip
          batch_size: the mini-batch size.
          train_transform: the transformations used by the sampler, they
            should be returned by the method get_train_transforms().
          num_workers: the total number of parallel workers for the samples.
          drop_last: it drops the last sample if the mini-batch cannot be
             aggregated, necessary for methods like DeepInfomax.
        Returns:
          train_loader: The loader that can be used a training time.
        """
        self._check()
        act_dict = self.get_act_dict()
        if (self.dataset == 'animalkingdom'):
            train_data = AnimalKingdom(self.path, act_dict, total_length=self.total_length, transform=train_transform,
                                       mode='train')
            sampler = RandomSampler(train_data, num_samples=2500)
            shuffle = False
        elif (self.dataset == 'baboonland'):
            train_data = BaboonLandDataset(self.path, act_dict, total_length=self.total_length,
                                           transform=train_transform,
                                           random_shift=False,
                                           mode='train')
            sampler = RandomSampler(train_data, num_samples=2500)
            shuffle = False
        elif (self.dataset == 'mammalnet'):
            train_data = MammalNet(self.path, act_dict, total_length=self.total_length,
                                   transform=train_transform,
                                   random_shift=False,
                                   mode='train')
            sampler = RandomSampler(train_data, num_samples=2500)
            shuffle = False
        else:
            raise Exception("[ERROR] The dataset " + str(self.dataset) + " is not supported!")

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers,
                                  sampler=sampler, pin_memory=False, drop_last=drop_last)
        return train_loader

    def get_test_loader(self, test_transform, drop_last=False):
        """Returns the test loader for each dataset.
           If a new method or dataset is added, this method should by modified
           accordingly.
        Args:
          path: disk location of the dataset.
          dataset: the name of the dataset.
          total_length: the number of frames in a video clip
          batch_size: the mini-batch size.
          train_transform: the transformations used by the sampler, they
            should be returned by the method get_train_transforms().
          num_workers: the total number of parallel workers for the samples.
          drop_last: it drops the last sample if the mini-batch cannot be
             aggregated, necessary for methods like DeepInfomax.
        Returns:
          train_loader: The loader that can be used a training time.
        """
        self._check()
        act_dict = self.get_act_dict()
        if (self.dataset == 'animalkingdom'):
            test_data = AnimalKingdom(self.path, act_dict, total_length=self.total_length, transform=test_transform,
                                      mode='val')
        if (self.dataset == 'baboonland'):
            test_data = BaboonLandDataset(self.path, act_dict, total_length=self.total_length, transform=test_transform,
                                          mode='val')
        elif (self.dataset == 'mammalnet'):
            test_data = MammalNet(self.path, act_dict, total_length=self.total_length, transform=test_transform,
                                  mode='test')
        else:
            raise Exception("[ERROR] The dataset " + str(self.dataset) + " is not supported!")
        sampler = DistributedSampler(test_data, shuffle=False) if self.distributed else None
        shuffle = False
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers,
                                 sampler=sampler, pin_memory=True, drop_last=drop_last)
        return test_loader
