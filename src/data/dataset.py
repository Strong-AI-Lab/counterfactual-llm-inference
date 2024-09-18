
import abc
from typing import List, Union, Iterator, Optional
import os
import json

class TextDataset(abc.ABC):

    def __init__(self, path : Union[str,List[str]]) -> None:
        self.data = None
        self._load_data(path)

    def _load_data(self, path : Union[str,List[str]]) -> None:
        if isinstance(path, str):
            path = [path]

        self.data = []
        for p in path:
            if os.path.isdir(p):
                self.data += self._load_data_from_dir(p)
            else:
                self.data.append(self._load_data_from_file(p))

    @abc.abstractmethod
    def _load_data_from_dir(self, path : str) -> List[str]:
        pass

    @abc.abstractmethod
    def _load_data_from_file(self, path : str) -> str:
        pass

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> str:
        return self.data[idx]

    def __iter__(self) -> Iterator[str]:
        return iter(self.data)
    


class CacheTextDataset(TextDataset):

    def _load_data_from_dir(self, path: str) -> List[str]:
        data = []
        for file in os.listdir(path):
            if file.endswith(".txt"):
                data.append(self._load_data_from_file(os.path.join(path, file)))
        
        return data
    
    def _load_data_from_file(self, path: str) -> str:
        name = os.path.basename(path)[0:-5]
        with open(path, 'r') as f:
            return name, f.read()



class JsonTextDataset(TextDataset):

    def _load_data_from_dir(self, path : str) -> List[str]:
        data = []
        for file in os.listdir(path):
            if file.endswith(".json"):
                d = self._load_data_from_file(os.path.join(path, file))
                if d is not None:
                    data.append(d)
        
        return data
    
    def _load_data_from_file(self, path : str) -> Optional[str]:
        with open(path, 'r') as f:
            j_f = json.load(f)
        
        if 'summary' in j_f:
            name = os.path.basename(path)[0:-5]
            return name, j_f['summary']
        else:
            return None
        


DATASETS = {
    'json' : JsonTextDataset,
    'cache' : CacheTextDataset
}