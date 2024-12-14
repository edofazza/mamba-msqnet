import os
import csv
import glob
import numpy as np
from PIL import Image
from torch.utils import data
from itertools import compress
import pandas as pd


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return self._data[2]
    

class ActionDataset(data.Dataset):
    def __init__(self, total_length):
        self.total_length = total_length
        self.video_list = []
        self.random_shift = False

    def _sample_indices(self, num_frames):
        if num_frames <= self.total_length:
            indices = np.linspace(0, num_frames - 1, self.total_length, dtype=int)
        else:
            ticks = np.linspace(0, num_frames, self.total_length + 1, dtype=int)
            if self.random_shift:
                indices = ticks[:-1] + np.random.randint(ticks[1:] - ticks[:-1])
            else:
                indices = ticks[:-1] + (ticks[1:] - ticks[:-1]) // 2
        return indices
    
    @staticmethod
    def _load_image(directory, image_name):
        return [Image.open(os.path.join(directory, image_name)).convert('RGB')]
    
    def __getitem__(self, index):
        record = self.video_list[index]
        image_names = self.file_list[index]
        indices = self._sample_indices(record.num_frames)
        return self._get(record, image_names, indices)
    
    def __len__(self):
        return len(self.video_list)
        

class AnimalKingdom(ActionDataset):
    def __init__(self, path, act_dict, total_length=12, transform=None, random_shift=False, mode='train'):
        self.path = path
        self.total_length = total_length
        self.transform = transform
        self.random_shift = random_shift
        self.mode = mode
        self.anno_path = os.path.join(self.path, 'action_recognition', 'annotation', mode + '_light.csv')
        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        try:
            self.video_list, self.file_list = self._parse_annotations()
        except OSError:
            print('ERROR: Could not read annotation file "{}"'.format(self.anno_path))
            raise

    def _parse_annotations(self):
        video_list = []
        file_list = []
        with open(self.anno_path) as f:
            reader = csv.DictReader(f, delimiter=';')
            for row in reader:
                ovid = row['video_id']
                labels = row['labels']
                path = os.path.join(self.path, 'action_recognition', 'dataset', 'image', ovid)
                files = sorted(os.listdir(path))
                file_list += [files]
                count = len(files)
                labels = [int(l) for l in labels.split(',')]
                video_list += [VideoRecord([path, count, labels])]
        return video_list, file_list

    def _get(self, record, image_names, indices):
        images = list()
        for idx in indices:
            try:
                img = self._load_image(record.path, image_names[idx])
            except:
                print('ERROR: Could not read image "{}"'.format(os.path.join(record.path, image_names[idx])))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(img)
        process_data = self.transform(images)
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
        label = np.zeros(self.num_classes)  # need to fix this hard number
        label[record.label] = 1.0
        return process_data, label
        

class BaboonLandDataset(ActionDataset):
    def __init__(self, path, act_dict, total_length=12, transform=None, random_shift=False, mode='train'):
        """
        Initialize the BaboonLand dataset.

        Args:
            path (str): Base path to the dataset
            act_dict (dict): Dictionary mapping action labels to integer indices
            total_length (int, optional): Number of frames to sample. Defaults to 12.
            transform (callable, optional): Optional transform to be applied on a list of images. Defaults to None.
            random_shift (bool, optional): Whether to randomly shift frame sampling. Defaults to False.
            mode (str, optional): Dataset mode, either 'train' or 'val'. Defaults to 'train'.
        """
        # Call the parent class constructor
        super().__init__(total_length)

        self.path = path
        self.transform = transform
        self.random_shift = random_shift
        self.mode = mode

        # Construct annotation path
        self.anno_path = os.path.join(self.path, f'annotation/{mode}.csv')

        self.act_dict = act_dict
        self.num_classes = len(act_dict)

        try:
            self.video_list, self.file_list = self._parse_annotations()
        except OSError:
            print(f'ERROR: Could not read annotation file "{self.anno_path}"')
            raise

    def _parse_annotations(self):
        """
        Parse the annotation CSV file and prepare video records.

        Returns:
            Tuple of video_list and file_list
        """
        # Read the annotation CSV
        df = pd.read_csv(self.anno_path, delimiter=' ')
        video_list = []
        file_list = []

        # Group by unique video_id to handle videos with multiple frames
        for video_id, video_group in df.groupby('video_id'):
            # Sort frames by frame_id to ensure correct order
            video_group = video_group.sort_values('frame_id')

            # Get unique video path (assuming first row represents the video path)
            video_path = os.path.dirname(video_group['path'].iloc[0])

            # Get all frame filenames for this video
            frame_files = video_group['path'].apply(os.path.basename).tolist()

            # Prepare labels
            labels = video_group['labels'].values
            unique_labels = np.unique(labels)

            # Create one-hot encoded labels for all frames
            num_frames = len(frame_files)
            label_matrix = np.zeros((num_frames, self.num_classes), dtype=bool)
            for i, label in enumerate(labels):
                label_matrix[i, label] = 1

            # Add to lists
            file_list.append(frame_files)
            video_list.append(VideoRecord(
                [
                    os.path.join(self.path, 'dataset', 'image', video_path),
                    num_frames,
                    label_matrix
                ]
            ))

        return video_list, file_list

    def _get(self, record, image_names, indices):
        """
        Retrieve and process images for the given indices.

        Args:
            record (VideoRecord): Video record containing path and metadata
            image_names (list): List of image filenames
            indices (np.ndarray): Indices of frames to retrieve

        Returns:
            Tuple of processed images and corresponding labels
        """
        images = []
        for idx in indices:
            try:
                img = self._load_image(record.path, image_names[idx])
            except OSError:
                print(f'ERROR: Could not read image "{record.path}/{image_names[idx]}"')
                print(f'Invalid indices: {indices}')
                raise
            images.extend(img)

        # Apply transformations
        process_data = self.transform(images)
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])

        # Get labels - take the max across sampled frames (any frame with the label)
        label = record.label[indices].any(0).astype(np.float32)

        return process_data, label