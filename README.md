# FAFNet

Code for the paper: "FAFNet: Pansharpening via Frequency-Aware Fusion Network with Explicit Similarity Constraints", IEEE TGRS, 2023

If you use our code, please cite the paper.

## Dependency
python  3.6 - 3.8 .      
pytorch  1.6.0-1.8.0.


# Directory Structure

│  README.md
│  args_parser.py                                    
│  data.py                                             
│  modelv12_2_h_all.py                   // FAFNet model. **Note: You should comment out the codes about HFS loss when testing.
│  mylib.py                                    // Some useful functions
│  test_reduced.py                         //  Test codes of FAFNet at reduced resolution
│  test_full.py                 // Test codes of FAFNet at  full resolution
│  trainv12_2.py                      // Training codes
│
├─image     // Model Architecture
│      FAFNet13    
├─DWT_IDWT          // DWT and IDWT layers. Refer to "Wavelet Integrated CNNs for Noise-Robust Image Classification, CVPR 2020" for more details.  
│  │  __init__.py
│  │  DWT_IDWT_Functions.py   
│  │  DWT_IDWT_layer.py
│
├─trained_model     
│      model_epoch1999_wv4.pth    // Pretrained model for WV4 dataset
│
└─WV4   // dataset
    ├─test      //  test images
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

Construct the training data according to the Wald's protocol

Put the training ms data in "WV4\train\\ms\\**1.mat" 

Put its corresponding pan data  in "WV4\train\\pan\\**1.mat" with the same file name

Download the pretrained model in the following link:
Link: https://pan.baidu.com/s/1NRxsBKCZfMDZGmYqBBa73Q 
code: qrp7



# Citation

If you find this code helpful, please kindly cite:



