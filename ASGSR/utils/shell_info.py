# read some info from .sh file
def read_num_workers(sh_path):
    with open(sh_path) as f:
        for line in f:
            if line.startswith('#SBATCH --ntasks-per-node'):
                process = line.split('#')[1]  # #SBATCH --ntasks-per-node=16
                process = process.strip()  # #SBATCH --ntasks-per-node=16
                process = int(process.split('=')[1])
                return process
