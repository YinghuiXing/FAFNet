# FAFNet

Code for the paper: "FAFNet: Pansharpening via Frequency-Aware Fusion Network with Explicit Similarity Constraints", IEEE TGRS, 2023

If you use our code, please cite the paper.

## Dependency
python  3.6 - 3.8 .      
pytorch  1.6.0-1.8.0.


# Directory Structure

│  README.md  
│  args\_parser.py  
│  data.py  
│  data\_generate.py  
│  modelv12\_2\_h\_all.py      // FAFNet model. **Note: You should comment out the codes about HFS loss when testing.  
│  mylib.py                           // Some useful functions  
│  quality\_assessment.py     // Some useful functions  
│  README.md  
│ test\_reduced.py               // Test codes of FAFNet at reduced resolution  
│ test\_full.py                     // Test codes of FAFNet at full resolution  
│  trainv12\_2.py               // Training codes  
├─DWT\_IDWT          // DWT and IDWT layers. Refer to "Wavelet Integrated CNNs for Noise-Robust Image Classification, CVPR 2020" for more details.  
│  │  DWT\_IDWT\_Functions.py  
│  │  DWT\_IDWT\_layer.py  
│  │  \_\_init\_\_.py  
│    
│  
├─image           // Model Architecture  
│      FAFNet13.pdf  
│  
├─trained_model    
│      model\_epoch1999\_wv4.pth    // Pretrained model for WV4 dataset   
│  
└─WV4  
    ├─test  
    │  ├─ms  
    │  │      36.mat  
    │  │      39.mat  
    │  │
    │  └─pan  
    │          36.mat  
    │          39.mat  
    │
    ├─train  
    │  ├─ms  
    │  └─pan  
    └─val  
        ├─ms  
        └─pan  


# Get Started 

Prepare data following the provided examples:<br>

Crop the large scale remote sensing images into small patches, and note that paired MS and PAN patches should match to each other.

Construct the training data according to the Wald's protocol

Put the training ms data in "WV4\train\\ms\\**1.mat" 

Put its corresponding pan data  in "WV4\train\\pan\\**1.mat" with the same file name

Download the pretrained model in the following link:

Link: https://pan.baidu.com/s/1NRxsBKCZfMDZGmYqBBa73Q   
code: qrp7



# Citation

If you find this code helpful, please kindly cite:

@article{xing2023pansharpening,  
  title={Pansharpening via Frequency-Aware Fusion Network with Explicit Similarity Constraints},    
  author={Xing, Yinghui and Zhang, Yan and He, Houjun and Zhang, Xiuwei and Zhang, Yanning},  
  journal={IEEE Transactions on Geoscience and Remote Sensing},  
  year={2023},  
  publisher={IEEE}  
}



