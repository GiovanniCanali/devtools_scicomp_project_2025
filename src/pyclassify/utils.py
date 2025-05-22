import os
import yaml
from line_profiler import profile
import numpy as np
from pyclassify._compiled import _distance_numba_compiled

@profile
def distance(point1: list[float], point2: list[float]) -> float:
    return sum((p1 - p2)**2 for p1, p2 in zip(point1, point2))

@profile
def distance_numpy(point1: np.ndarray, point2: np.ndarray) -> float:
    d = point1-point2
    return np.dot(d,d)

@profile
def distance_numba(point1: np.ndarray, point2: np.ndarray) -> float:
    return _distance_numba_compiled(point1, point2)

@profile
def majority_vote(neighbors: list[int]) -> int:
    return max(set(neighbors), key=neighbors.count)

def read_config(file):
   filepath = os.path.abspath(f'{file}.yaml')
   with open(filepath, 'r') as stream:
      kwargs = yaml.safe_load(stream)
   return kwargs

def read_file(file: str) -> tuple[list[list[float]], list[int]]:
    filepath = os.path.abspath(f'./{file}')
    features, labels = [], []
    with open(filepath, 'r') as stream:
        for line in stream:
            data = line.strip().split(',')

            features.append([float(x) for x in data[:-1]])
            label = data[-1].lower()
            
            if label.isdigit() and int(label) in [0, 1]:
                labels.append(int(label))
            else:
                labels.append(0 if data[-1][0] == 'b' else 1)

    return features, labels