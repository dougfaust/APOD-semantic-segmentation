# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {
    "data": {
        "train_dir": "data/train",
        "test_dir": "data/test",
        "data_transform": None,
        "image_size": 128,
        "load_with_info": True
    },
    "train": {
        "batch_size": 64,
        "num_workers": 1,
        "buffer_size": 1000,
        "epoches": 20,
        "val_subsplits": 5,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy"]
    },
    "model": {
        "input": [128, 128, 3],
        "up_stack": {
            "layer_1": 512,
            "layer_2": 256,
            "layer_3": 128,
            "layer_4": 64,
            "kernels": 3
        },
        "output": 3
    }
}
