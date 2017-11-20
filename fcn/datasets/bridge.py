import scipy.io
from .. import data

DATASET_BRIDGE_DIR = osp.expanduser('~/repos/bridgedegradationseg/dataset/')

class BridgeSegBase(chainer.dataset.DatasetMixin):

    class_names = np.array(['non-damage', 'damage'])

    def __init__(self, split='train'):
        self.split = split

        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(DATASET_BRIDGE_DIR, "{}.txt".format(split))
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(DATASET_BRIDGE_DIR, 'images/combined/', '{}.jpg'.format(did))
                lbl_file = osp.join(DATASET_BRIDGE_DIR, 'bridge_masks/combined/', '{}.png'.format(did))
                self.files[split].append({
                    'img' : img_file,
                    'lbl' : lbl_file,
                })

    def __len__(self):
        return len(self.files[self.split])

    def get_example(self, index):
        data_file = self.files[self.split][index]
        img_file = data_file['img']
        img = Image.open(img_file)
        lbl_file = self.files[self.split][index]
        lbl = Image.open(lbl_file)

        img = np.array(img, dtype=np.uint8)
        lbl = np.array(lbl, dtype=np.uint32)
        lbl = lbl/255
        return img, lbl

class BridgeSeg(BridgeSegBase):
    def __init__(self, split='train'):
       super(BridgeSeg, self).__init__(split=split) 


    @staticmethod
    def download():
        print("Not implemented yet (could be a good idea)")
