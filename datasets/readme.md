Acquire the various datasets and placed them in this folder to run evaluation on them.
Note that the high resolution dataset is not publicly available.

The expected directory structure is:
```
datasets/
├── Chugh/
│   ├── label.txt
│   └── <png eye images>
├── high_res_1000Hz/
│   ├── ss01/
│   │   └── <mp4 eye videos>
│   ├── ss02/
│   ├── ss03/
│   └── ss04/
├── high_res_500Hz/
│   ├── ss01/
│   │   └── <mp4 eye videos>
│   ├── ss02/
│   ├── ss03/
│   └── ss04/
├── openEDS2019_segmentation/
│   ├── test/
│   │   ├── images/
│   │   │   └── <png eye images>
│   │   └── labels/
│   │       └── <npy label files>
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   └── validation/
│       ├── images/
│       └── labels/
└── openEDS2020_partialSeg/
    ├── S_0/
    │   └── <png eye images and npy label files>
    ├── S_1/
    ├── ...
    ├── S_198/
    ├── S_199/
    ├── labels.txt
    └── test_sampleName_GT.pkl
```