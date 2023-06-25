import numpy as np
import torch
import copy
import pickle
from typing import Any, List, Union

IndexType = Union[slice, int, np.ndarray, List[int]]


def _create_placeholder(inst, size):
    if isinstance(inst, np.ndarray):
        return np.zeros(shape = [size, *inst.shape[1:]], dtype = np.float32)
    elif isinstance(inst, torch.Tensor):
        return torch.zeros(size = [size, *inst.size()[1:]], device = inst.device, dtype = inst.dtype)
    elif isinstance(inst, Batch):
        holder = Batch()
        for key, val in inst.items():
            holder[key] = _create_placeholder(val, size)
        return holder
    else:
        raise NotImplementedError



'''
    Data batch
    Reference: https://tianshou.readthedocs.io/en/master/_modules/tianshou/data/batch.html#Batch
'''
class Batch(object):
    def __init__(self, **data_dict) -> None:
        for key, val in data_dict.items():
            if isinstance(val, (np.ndarray, torch.Tensor, Batch)):
                self.__dict__[key] = val
            else:
                if isinstance(val, dict):
                    # To load "Batch" from an existing dictionary.
                    self.__dict__[key] = Batch(**val)
                else:
                    raise TypeError('Not supported data structure type!')


    def keys(self):
        return self.__dict__.keys()
    

    def values(self):
        return self.__dict__.values()
    

    def items(self):
        return self.__dict__.items()
    

    def __getstate__(self):
        state = {}
        for key, obj in self.items():
            if isinstance(obj, Batch):
                state[key] = obj.__getstate__()
            else:
                state[key] = obj
        return state
    

    def __setstate__(self, state):
        self.__init__(**state)


    def __getitem__(self, ptr : Union[str, IndexType]):
        if isinstance(ptr, str):
            # Access with a key string
            return self.__dict__[ptr]
        else:
            # Access with index
            if len(self.keys()) > 0:
                new_batch = Batch()
                for key, val in self.items():
                    new_batch.__dict__[key] = val[ptr]
                return new_batch
            else:
                raise IndexError('Cannot access the empty batch!')
    

    def __setitem__(self, ptr : Union[str, IndexType], obj : "Batch"):
        if isinstance(ptr, str):
            self.__dict__[ptr] = obj
            return
        
        if not set(obj.keys()).issubset(self.__dict__.keys()):
            raise KeyError('Not existing key!')
        for key, val in obj.items():
            try:
                self.__dict__[key][ptr] = obj[key]
            except KeyError:
                if isinstance(val, Batch):
                    self.__dict__[key][ptr] = Batch()
                else:
                    # np.ndarray or torch.Tensor
                    self.__dict__[key][ptr] = 0
            

    def __repr__(self) -> str:
        res_str = []
        for key, obj in sorted(self.items()):
            if isinstance(obj, np.ndarray):
                res_str.append(f'{key}=Array{obj.shape}')
            elif isinstance(obj, torch.Tensor):
                res_str.append(f'{key}=Tensor{tuple(obj.size())}')
            else:
                res_str.append(f'{key}={str(obj)}')
        res_str = ', '.join(res_str)
        return f'Batch({res_str})'
    

    def __len__(self):
        lens = []
        for val in self.values():
            lens.append(len(val))
        return min(lens) if len(lens) > 0 else 0


    def to_numpy(self):
        for key, obj in self.items():
            if isinstance(obj, torch.Tensor):
                self.__dict__[key] = obj.cpu().numpy()
            elif isinstance(obj, Batch):
                obj.to_numpy()


    def float(self):
        for key, obj in self.items():
            if isinstance(obj, np.ndarray):
                self.__dict__[key] = obj.astype(np.float32)
            elif isinstance(obj, torch.Tensor):
                self.__dict__[key] = obj.float()
            elif isinstance(obj, Batch):
                obj.float()


    def to_torch(self, device : str = 'cpu'):
        for key, obj in self.items():
            if isinstance(obj, np.ndarray):
                self.__dict__[key] = torch.from_numpy(obj).to(torch.device(device))
            elif isinstance(obj, Batch):
                obj.to_torch(device)


    def is_empty(self, recursive = False):
        if len(self.keys()) == 0:
            return True
        else:
            if not recursive:
                return False
            else:
                return np.all([obj.is_empty(recursive) if isinstance(obj, Batch) else False for obj in self.values()])


    def cat_(self, batches : List["Batch"]):
        # Currently assume the given batches are all non-empty and have same keys
        all_batch_keys = [set(b.keys()) for b in batches]
        keys_union = set.union(*all_batch_keys)
        assert all(map(lambda x: x == keys_union, all_batch_keys))

        # Batch self can be either empty or non-empty (if non-empty, then should have the same keys)
        if not self.is_empty():
            assert set(self.keys()) == keys_union
            batches.insert(0, self)

        for key in keys_union:
            key_data = [b[key] for b in batches]
            key_data_type = [type(kd) for kd in key_data]

            if all(map(lambda x: x == torch.Tensor, key_data_type)):
                self.__dict__[key] = torch.cat(key_data)
            elif all(map(lambda x: x == np.ndarray, key_data_type)):
                self.__dict__[key] = np.concatenate(key_data)
            elif all(map(lambda x: x == Batch, key_data_type)):
                holder = Batch()
                holder.cat_(key_data)
                self.__dict__[key] = holder
            else:
                raise KeyError(f'Fail to concatenate key "{key}".')


    @staticmethod
    def cat(batches : List["Batch"]):
        res = Batch()
        res.cat_(batches)
        return res
    

    def stack_(self, batches : List["Batch"], dim):
        # Currently assume the given batches are all non-empty and have same keys
        all_batch_keys = [set(b.keys()) for b in batches]
        keys_union = set.union(*all_batch_keys)
        assert all(map(lambda x: x == keys_union, all_batch_keys))

        # Batch self can be either empty or non-empty (if non-empty, then should have the same keys)
        if not self.is_empty():
            assert set(self.keys()) == keys_union
            batches.insert(0, self)

        for key in keys_union:
            key_data = [b[key] for b in batches]
            key_data_type = [type(kd) for kd in key_data]

            if all(map(lambda x: x == torch.Tensor, key_data_type)):
                self.__dict__[key] = torch.stack(key_data, dim = dim)
            elif all(map(lambda x: x == np.ndarray, key_data_type)):
                self.__dict__[key] = np.stack(key_data, axis = dim)
            elif all(map(lambda x: x == Batch, key_data_type)):
                holder = Batch()
                holder.stack_(key_data, dim = dim)
                self.__dict__[key] = holder
            else:
                raise KeyError(f'Fail to stack key "{key}".')


    @staticmethod
    def stack(batches : List["Batch"], dim = 0):
        res = Batch()
        res.stack_(batches, dim = dim)
        return res
    

    def split(self, batch_size = 256, shuffle = True):
        length = len(self)
        indices = np.random.permutation(length) if shuffle else np.arange(length)
        start_idx = 0
        while start_idx < length:
            yield self[indices[start_idx:start_idx + batch_size]]
            start_idx += batch_size



'''
    Replay buffer
'''
class BaseBuffer(object):
    necessary_keys = set(['obs', 'next_obs', 'act', 'rew', 'done'])

    def __init__(self, capacity = 100000, seed = None) -> None:
        self.capacity = capacity
        self.reset()
        np.random.seed(seed)


    def reset(self):
        self.buffer = Batch()
        self.curr_index = 0
        self.curr_size = 0


    @property
    def size(self):
        return self.curr_size


    def add(self, batch):
        raise NotImplementedError


    def sample(self, batch_size = 256, duplicate = False):
        assert not (batch_size > self.size and not duplicate)
        indices = np.random.choice(self.size, batch_size, replace = duplicate)
        return self.buffer[indices]
    

    def get_whole_buffer(self, shuffle = False):
        indices = np.random.permutation(self.size) if shuffle else np.arange(self.size)
        return self.buffer[indices]
    

    def extend_buffer(self, new_capacity):
        if self.curr_index < self.size:
            self.curr_index = self.capacity
        self.capacity = new_capacity

        # Copy the buffer
        old_buffer = copy.deepcopy(self.buffer)
        self.buffer = _create_placeholder(old_buffer, self.capacity)
        self.buffer[:self.curr_index] = old_buffer


    def save_file(self, file_path):
        with open(file_path, 'wb') as f:
            save_dict = {}
            save_dict['buffer'] = self.buffer
            save_dict['curr_index'] = self.curr_index
            save_dict['capacity'] = self.capacity
            pickle.dump(save_dict, f)
            print(f'{type(self).__name__}: Save to {file_path}!')


    def load_file(self, file_path):
        with open(file_path, 'rb') as f:
            sd = pickle.load(f)
            self.buffer = sd['buffer']
            self.curr_index = sd['curr_index']
            self.capacity = sd['capacity']
            print(f'{type(self).__name__}: Load from {file_path}!')



class ReplayBuffer(BaseBuffer):
    def add(self, batch : Batch):
        assert self.necessary_keys.issubset(set(batch.keys()))
        
        for single_batch in batch.split(1, False):
            try:
                self.buffer[self.curr_index] = single_batch
            except KeyError:
                if self.buffer.is_empty():
                    self.buffer = _create_placeholder(batch, self.capacity)
                    self.buffer[self.curr_index] = single_batch
                else:
                    raise NotImplementedError
            self.curr_index = (self.curr_index + 1) % self.capacity
            if self.curr_size < self.capacity:
                self.curr_size += 1






if __name__ == '__main__':
    batch = Batch(
        obs = np.random.randn(3, 4),
        next_obs = np.random.randn(3, 4),
        act = np.random.randn(3, 2),
        rew = np.random.randn(3),
        done = np.random.randn(3)
    )

    buffer = ReplayBuffer(5)
    for _ in range(3):
        buffer.add(batch)
        print(buffer.buffer.done)

    buffer.extend_buffer(10)
    for _ in range(3):
        buffer.add(batch)
        print(buffer.buffer.done)

    # buffer.save_file('test.pkl')
    # buffer.load_file('test.pkl')
    # print(buffer.curr_index, buffer.capacity, buffer.buffer)