from secrets import randbelow
from torch.utils import data
import  torchvision.transforms.functional as TF
import torch
import numpy as np
import pickle 
import os    
       
from multiprocessing import Process, Manager   


class Utterances(data.Dataset):
    """Dataset class for the Utterances dataset."""

    def __init__(self, root_dir, len_crop):
        """Initialize and preprocess the Utterances dataset."""
        self.root_dir = root_dir
        self.len_crop = len_crop
        self.step = 10
        
        metaname = os.path.join(self.root_dir, "train.pkl")
        meta = pickle.load(open(metaname, "rb"))
        
        """Load data using multiprocessing"""
        manager = Manager()
        meta = manager.list(meta)
        dataset = manager.list(len(meta)*[None])  
        processes = []
        for i in range(0, len(meta), self.step):
            p = Process(target=self.load_data, 
                        args=(meta[i:i+self.step],dataset,i))  
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            
        self.train_dataset = list(dataset)
        self.num_tokens = len(self.train_dataset)
        
        print('Finished loading the dataset...')
        
        
    def load_data(self, submeta, dataset, idx_offset):  
        for k, sbmt in enumerate(submeta):    
            uttrs = len(sbmt)*[None]
            for j, tmp in enumerate(sbmt):
                if j < 2:  # fill in speaker id and embedding
                    uttrs[j] = tmp
                else: # load the mel-spectrograms
                    uttrs[j] = np.load(os.path.join(self.root_dir, tmp))
            dataset[idx_offset+k] = uttrs
                   
        
    def __getitem__(self, index):
        # pick a random speaker
        dataset = self.train_dataset 
        list_uttrs = dataset[index]
        emb_org = list_uttrs[1]
        
        # pick random uttr with random crop
        a = np.random.randint(2, len(list_uttrs))
        tmp = list_uttrs[a]
        if tmp.shape[0] < self.len_crop:
            len_pad = self.len_crop - tmp.shape[0]
            uttr = np.pad(tmp, ((0,len_pad),(0,0)), 'constant')
        elif tmp.shape[0] > self.len_crop:
            left = np.random.randint(tmp.shape[0]-self.len_crop)
            uttr = tmp[left:left+self.len_crop, :]
        else:
            uttr = tmp
        
        return uttr, emb_org
    

    def __len__(self):
        """Return the number of spkrs."""
        return self.num_tokens
    


class Specs(data.Dataset):
    """Dataset class for the Spectrograms dataset."""

    def __init__(self, root_dir, len_crop=None):
        """Initialize and preprocess the Utterances dataset."""
        self.root_dir = root_dir
        self.len_crop = len_crop

        # Get a list of all files with "feats" in the name
        # in the root directory
        self.files = []
        for r, d, f in os.walk(self.root_dir):
            for file in f:
                if '.npy' in file and 'feats' in file:
                    self.files.append(os.path.join(r, file))
        
        # metaname = os.path.join(self.root_dir, "train.pkl")
        # meta = pickle.load(open(metaname, "rb"))
        
        """Load data using multiprocessing"""
        manager = Manager()
        dataset = manager.list(len(self.files)*[None])  
        processes = []
        for i in range(0, len(self.files)):
            p = Process(target=self.load_data, 
                        args=(dataset,i))  
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            
        self.train_dataset = list(dataset)
        self.num_tokens = len(self.train_dataset)
        
        print('Finished loading the dataset...')
        
        
    def load_data(self, dataset, idx_offset):  
        dataset[idx_offset] = np.load(self.files[idx_offset])
                   
        
    def __getitem__(self, index):
        spec = self.train_dataset[index]

        if self.len_crop is None:
            return spec
        
        # Crop or pad to shape
        tmp = spec
        if tmp.shape[0] < self.len_crop:
            len_pad = self.len_crop - tmp.shape[0]
            spec = np.pad(tmp, ((0,len_pad),(0,0)), 'constant')
        elif tmp.shape[0] > self.len_crop:
            left = np.random.randint(tmp.shape[0]-self.len_crop)
            spec = tmp[left:left+self.len_crop, :]
        else:
            spec = tmp
        
        return spec
    

    def __len__(self):
        """Return the number of spkrs."""
        return self.num_tokens


class SpecsCombined(data.Dataset):
    """Dataset class for the Spectrograms dataset."""

    def __init__(self, root_dir, len_crop=None, random_flip=False, random_crop=False):
        """Initialize and preprocess the Utterances dataset."""
        self.root_dir = root_dir
        self.len_crop = len_crop
        self.random_flip = random_flip
        self.random_crop = random_crop

        # Get a list of all files with "feats" in the name
        # in the root directory
        self.files = []
        for r, d, f in os.walk(self.root_dir):
            for file in f:
                if '.npy' in file and 'feats' in file and 'accom' in file:
                    self.files.append(os.path.join(r, file))
        
        # metaname = os.path.join(self.root_dir, "train.pkl")
        # meta = pickle.load(open(metaname, "rb"))
        
        """Load data using multiprocessing"""
        manager = Manager()
        dataset = manager.list(len(self.files)*[None])  
        processes = []
        for i in range(0, len(self.files)):
            p = Process(target=self.load_data, 
                        args=(dataset,i))  
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
            
        self.train_dataset = list(dataset)

        # Filter the dataset so all spectrograms have the same length
        tmp = []
        tmp_files = []
        for i in range(0, len(self.train_dataset)):
            accom, spec = self.train_dataset[i]
            if accom.shape[0] == spec.shape[0]:
                tmp.append(self.train_dataset[i])
                tmp_files.append(self.files[i])
        self.train_dataset = tmp
        self.files = tmp_files

        self.num_tokens = len(self.train_dataset)
        
        print('Finished loading the dataset...')
        
        
    def load_data(self, dataset, idx_offset):
        accom_path = self.files[idx_offset]
        vocals_path = accom_path.replace('accom', 'vocals')
        dataset[idx_offset] = (np.load(accom_path), np.load(vocals_path))


    def crop_or_pad_spec(self, spec, left):
        tmp = spec
        if tmp.shape[0] < self.len_crop:
            len_pad = self.len_crop - tmp.shape[0]
            spec = np.pad(tmp, ((0,len_pad),(0,0)), 'constant')
        elif tmp.shape[0] > self.len_crop:
            spec = tmp[left:left+self.len_crop, :]
        else:
            spec = tmp
        return spec


    def dual_random_resized_crop(self, x1, x2, crop_range=0.45):
        assert x1.shape == x2.shape

        x1 = x1.unsqueeze(0)
        x2 = x2.unsqueeze(0)

        _, h, w = x1.shape

        top = h * (crop_range * torch.rand(1))
        left = w * (crop_range * torch.rand(1))
        bottom = h * (1 - (crop_range * torch.rand(1)))
        right = w * (1 - (crop_range * torch.rand(1)))

        x1_r = TF.resized_crop(x1, int(top), int(left), int(bottom), int(right), (h, w))
        x2_r = TF.resized_crop(x2, int(top), int(left), int(bottom), int(right), (h, w))

        x1_r = x1_r.squeeze(0)
        x2_r = x2_r.squeeze(0)

        return x1_r, x2_r
                   
        
    def __getitem__(self, index):
        accom_spec, vocals_spec = self.train_dataset[index]

        if self.len_crop is None:
            return accom_spec, vocals_spec
        
        # Crop or pad to shape
        if accom_spec.shape[0] < self.len_crop:
            left = 0
        else:
            left = np.random.randint(accom_spec.shape[0]-self.len_crop)
        accom_spec = self.crop_or_pad_spec(accom_spec, left)
        vocals_spec = self.crop_or_pad_spec(vocals_spec, left)

        if self.random_flip and np.random.rand() > 0.5:
            accom_spec = TF.hflip(torch.from_numpy(accom_spec))
            vocals_spec = TF.hflip(torch.from_numpy(vocals_spec))
        else:
            accom_spec = torch.from_numpy(accom_spec)
            vocals_spec = torch.from_numpy(vocals_spec)

        if self.random_crop:
            accom_spec, vocals_spec = self.dual_random_resized_crop(accom_spec, vocals_spec)
        
        return accom_spec, vocals_spec
    

    def __len__(self):
        """Return the number of spkrs."""
        return self.num_tokens

    

def get_loader(root_dir, batch_size=16, len_crop=880, num_workers=4):
    """Build and return a data loader."""
    
    dataset = Specs(root_dir, len_crop)
    
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn)
    return data_loader


def get_loader_combined(root_dir, batch_size=16, len_crop=128, num_workers=4, random_flip=False, random_crop=False):
    """Build and return a data loader."""
    
    dataset = SpecsCombined(root_dir, len_crop, random_flip=random_flip, random_crop=random_crop)

    # Split dataset into training and validation sets
    train_dataset, val_dataset = torch.utils.data.random_split(dataset,
        [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])
    
    worker_init_fn = lambda x: np.random.seed((torch.initial_seed()) % (2**32))
    train_loader = data.DataLoader(dataset=train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn)
    val_loader = data.DataLoader(dataset=val_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True,
                                  worker_init_fn=worker_init_fn)

    return train_loader, val_loader


if __name__ == "__main__":
    loader, val_loader = get_loader_combined("~/Data/ts_segments_combined", batch_size=16, len_crop=860, random_crop=True)
    # print(next(iter(loader))[0].shape)
    for data in loader:
        print(data[0].shape)
