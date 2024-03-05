import torch 
import time
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Matrix multiplication')
    # Accepts a list of GPU IDs instead of the number of GPUs
    parser.add_argument('--gpu_id', type=int, required=True, help='List of GPU IDs to use')
    parser.add_argument('--size', help='matrix size', required=True, type=int)
    parser.add_argument('--interval', help='sleep interval', required=True, type=float)
    args = parser.parse_args()
    return args

def matrix_multiplication(args):
    a_list, b_list, result = [], [], []
    size = (args.size, args.size)

    # Use the specified GPU IDs
    gpu_id = args.gpu_id
    device = torch.device(f'cuda:{gpu_id}')
    a_list.append(torch.rand(size, device=device))
    b_list.append(torch.rand(size, device=device))
    result.append(torch.rand(size, device=device))

    while True:
        for i, gpu_id in enumerate([args.gpu_id]):
            result[i] = torch.matmul(a_list[i], b_list[i])
        time.sleep(args.interval)

if __name__ == "__main__":
    args = parse_args()
    matrix_multiplication(args)
