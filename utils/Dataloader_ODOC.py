from torch.utils.data import Dataset
import h5py
from torchvision.transforms import functional
import torchvision.transforms as transforms


class ODOC(Dataset):
    """ ODOC Dataset """
    def __init__(self, base_dir=None, split='test'):
        self._base_dir = base_dir
        self.sample_list = []
        test_path = self._base_dir + '/' + str(split) + '.list'
        with open(test_path, 'r') as f:
            self.image_list = f.readlines()

        self.image_list = [item.replace('\n', '') for item in self.image_list]

        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        h5f = h5py.File(self._base_dir + '/h5py_all' + '/'+image_name, 'r')
        image = self.test_transform(h5f['img'][:])
        label = functional.to_tensor(h5f['mask'][:])
        con_gau = functional.to_tensor(h5f['con_gau'][:])
        sample = {'img': image, 'mask': label, 'con_gau': con_gau}
        return sample, image_name


