#!/usr/bin/env python3
import torch
import csv
import os

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

    # Read comma-separated CSV file in /data/samples.csv
    print("Reading /data/samples.csv")
    with open('/data/samples.csv', mode='r') as file:
        reader = csv.reader(file)
        data = [torch.tensor(list(map(float, row))) for row in reader]

    # Sum each row using torch
    sums = [row.sum() for row in data]

    # Ensure the results directory exists
    os.makedirs('/results', exist_ok=True)

    # Write the sum of each row to /results/sums.csv
    print("Writing /results/sums.csv")
    with open('/results/sums.csv', mode='w') as file:
        writer = csv.writer(file)
        for sum_value in sums:
            writer.writerow([sum_value.item()])
