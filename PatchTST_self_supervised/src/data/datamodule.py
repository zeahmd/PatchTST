import warnings

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler


class DataLoaders:
    def __init__(
        self,
        datasetCls,
        dataset_kwargs: dict,
        batch_size: int,
        workers: int=0,
        collate_fn=None,
        shuffle_train = False,
        shuffle_val = False
    ):
        super().__init__()
        self.datasetCls = datasetCls
        self.batch_size = batch_size
        
        if "split" in dataset_kwargs.keys():
            del dataset_kwargs["split"]
        self.dataset_kwargs = dataset_kwargs
        self.workers = workers
        self.collate_fn = collate_fn
        self.shuffle_train, self.shuffle_val = shuffle_train, shuffle_val
        self.label_distribution = {}
    
        self.train = self.train_dataloader()
        self.valid = self.val_dataloader()
        self.test = self.test_dataloader()        
 
        
    def train_dataloader(self):
        return self._make_dloader("train", shuffle=self.shuffle_train)


    def val_dataloader(self):        
        return self._make_dloader("val", shuffle=self.shuffle_val)


    def test_dataloader(self):
        return self._make_dloader("test", shuffle=False)


    def get_sampler(self, dataset, training_mode, split):
        if (training_mode == "alltrain" and split=='train') or (training_mode == "finetune" and split=='train'):
            # compute class weights
            label_list = torch.tensor(dataset.get_bis_list())
            class_sample_count = torch.tensor(
                [(label_list == t).sum() for t in torch.unique(label_list)]
            )
            weight = 1. / class_sample_count.float()
            samples_weight = torch.tensor([weight[t] for t in label_list])
            sampler = WeightedRandomSampler(
                weights=samples_weight,
                num_samples=len(samples_weight),
                replacement=True
            )
            return sampler
        else:
            return None


    def get_label_distribution(self, split):
        return self.label_distribution.get(split, None)


    def _make_dloader(self, split, shuffle=False):
        dataset = self.datasetCls(**self.dataset_kwargs, split=split)
        if self.dataset_kwargs.get('mode') in ['alltrain', 'finetune']:
            self.label_distribution[split] = dataset.get_label_distribution()
        sampler = self.get_sampler(dataset, self.dataset_kwargs.get('mode'), split)
        if len(dataset) == 0: return None
        return DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=self.batch_size,
            num_workers=self.workers,
            collate_fn=self.collate_fn,
            sampler=sampler
        )


    @classmethod
    def add_cli(self, parser):
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument(
            "--workers",
            type=int,
            default=6,
            help="number of parallel workers for pytorch dataloader",
        )


    def add_dl(self, test_data, batch_size=None, **kwargs):
        # check of test_data is already a DataLoader
        from ray.train.torch import _WrappedDataLoader
        if isinstance(test_data, DataLoader) or isinstance(test_data, _WrappedDataLoader): 
            return test_data

        # get batch_size if not defined
        if batch_size is None: batch_size=self.batch_size        
        # check if test_data is Dataset, if not, wrap Dataset
        if not isinstance(test_data, Dataset):
            test_data = self.train.dataset.new(test_data)        
        
        # create a new DataLoader from Dataset 
        test_data = self.train.new(test_data, batch_size, **kwargs)
        return test_data

    
