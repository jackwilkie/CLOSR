'''
Numpy normalisation functions which use bessels correction

Created on: 24/10/24
'''

import numpy as np
from numpy import ndarray
from typing import Tuple, Optional, Union

# -- z-normalisation
def calc_stats(x: ndarray, bessels_correction: bool = True) -> Tuple[float, float]:
    return x.mean(axis = 0), x.std(axis = 0, ddof = 1 if bessels_correction else 0)

def z_normalise(x: ndarray, mean: float, std: float):
    return (x - mean)/std

class ZNormaliser:
    def __init__(
        self,
        x_data: Optional[ndarray] = None,
        bessels_correction: bool = True,
    ) -> None:
        ''' Init z-normaliser, calculate training data statistics if provided
        '''
        self.bessels_correction = bessels_correction
        self.mean, self.std = calc_stats(x = x_data, bessels_correction = self.bessels_correction) if x_data is not None else (None, None)
        
    def fit(self, x_data: ndarray) -> None:
        self.mean, self.std = calc_stats(x = x_data, bessels_correction = self.bessels_correction)

    def normalise(self, x_data: ndarray) -> ndarray:
        if self.mean is None or self.std is None:
            raise ValueError('ERROR::: ZNormaliser statistics have not been fit!')
        return z_normalise(x = x_data, mean = self.mean, std = self.std)
    
# -- label encoder
def encode_labels(y_data: ndarray, class_zero: Optional[Union[str, int]] = None, offset: int = 0):
    ''' function to convert string labels to ints
    '''
    possible_labels = np.unique(y_data)
    label_mapping = {}
    
    if class_zero is not None and class_zero in possible_labels:
        label_mapping[class_zero] = 0
     
    for new_label in possible_labels:
        if not new_label in label_mapping: 
            label_mapping[new_label] = int(len(label_mapping) + offset)

    new_labels = np.zeros(len(y_data), dtype = np.int64)
    for i in range(len(y_data)):
        new_labels[i] = label_mapping[y_data[i]]
    return new_labels
