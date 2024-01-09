#!/usr/bin/env python3
import torch

if __name__ == '__main__':
    print("PyTorch Version:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA Version:", torch.version.cuda)
        print("CuDNN Version:", torch.backends.cudnn.version())
        print("CUDA Device Name:", torch.cuda.get_device_name(0))
        print("Number of CUDA Devices Available:", torch.cuda.device_count())
        print("Current CUDA Device Index:", torch.cuda.current_device())
    else:
        print("CUDA is not available. No GPU information can be displayed.")
