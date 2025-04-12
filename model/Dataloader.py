import numpy as np
import os
import pickle

class cifar10:
    def __init__(self, root_dir, batch_size = 1, normalize = True, n_valid = 0):
        """
        root_dir: 数据集所在根目录
        normalize: 是否进行归一化
        """
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.normalize = normalize
        self.n_valid = n_valid
        self.data, self.labels = self.load_data(Train = True)
        self.train_data, self.train_labels, self.valid_data, self.valid_labels = self.train_valid_split(shuffle = True)
        self.test_data, self.test_labels = self.load_data(Train = False)

    def load_data(self, Train = True):
        files = [f'data_batch_{i}' for i in range(1,6)] if Train else ['test_batch']
        data, labels = [], []
        for file in files:
            batch_path = os.path.join(self.root_dir, file)
            with open(batch_path, 'rb') as f:
                batch = pickle.load(f, encoding = 'bytes')
                data.append(batch[b'data'])
                labels.extend(batch[b'labels'])
            
        data = np.vstack(data).reshape(-1, 3072).astype(np.uint8)
        if self.normalize:
            data = data / 255.0
        return data, np.array(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
    def train_valid_split(self, shuffle = True):
        n_valid = self.n_valid
        indices = np.arange(len(self))
        if shuffle:
            np.random.shuffle(indices)
        valid_indices = np.array(indices[:n_valid], dtype = np.int64)
        train_indices = np.array(indices[n_valid:], dtype = np.int64)
        return self.data[train_indices], self.labels[train_indices], self.data[valid_indices], self.labels[valid_indices]

    def get_train_batches(self, shuffle = True):
        batch_size = self.batch_size
        n_train = len(self.train_labels)
        indices = np.arange(n_train)
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, n_train, batch_size):
            end_idx = min(start_idx + batch_size, len(self))
            batch_indices = indices[start_idx:end_idx]
            batch_indices = np.array(batch_indices, dtype=np.int64)
            yield self.train_data[batch_indices], self.train_labels[batch_indices]

    def get_valid_batches(self, shuffle = True):
        batch_size = self.batch_size
        n_valid = len(self.valid_labels)
        indices = np.arange(n_valid)
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, n_valid, batch_size):
            end_idx = min(start_idx + batch_size, len(self))
            batch_indices = indices[start_idx:end_idx]
            batch_indices = np.array(batch_indices, dtype=np.int64)
            yield self.valid_data[batch_indices], self.valid_labels[batch_indices]

    def get_test_batches(self, shuffle = True):
        batch_size = self.batch_size
        n_test = len(self.test_labels)
        indices = np.arange(n_test)
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, n_test, batch_size):
            end_idx = min(start_idx + batch_size, len(self))
            batch_indices = indices[start_idx:end_idx]
            batch_indices = np.array(batch_indices, dtype=np.int64)
            yield self.test_data[batch_indices], self.test_labels[batch_indices]

# if __name__ == '__main__':
#     data = cifar10(
#         root_dir = './cifar10',
#         batch_size = 4,
#         normalize = True,
#         n_valid = 8
#     )
#     print(f'Size of traindata: {data.__len__()}')

#     sample, label = data[0]
#     print(f'sample shape: {sample.shape}')
#     print(f'label type: {type(label)}')

#     train_batch_loader = data.get_train_batches()
#     print('train_data')
#     for i, (img, label) in enumerate(train_batch_loader):
#         print(f'batch{i}: shape={img.shape}, img = {img},label={label}')
#         if i == 1:
#             break

#     valid_batch_loader = data.get_valid_batches()
#     print('valid_data')
#     for i, (img, label) in enumerate(valid_batch_loader):
#         print(f'batch{i}: shape={img.shape},label={label}')
#         if i == 2:
#             break
#     test_batch_loader = data.get_test_batches()
#     print('test_data')
#     for i, (img, label) in enumerate(test_batch_loader):
#         print(f'batch{i}: shape={img.shape},label={label}')
#         if i == 2:
#             break