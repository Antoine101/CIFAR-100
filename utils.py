import torch
import multiprocessing


def args_interpreter(args):

    print(f"Accelerator: {args.accelerator}")

    if args.devices.isdigit():
        args.devices = int(args.devices)

    n_cpus = multiprocessing.cpu_count()

    # Print the number and names of GPUs used
    if args.accelerator == "gpu":
        n_gpus = torch.cuda.device_count()
        if args.devices == "auto":
            print(f"Using all {n_gpus} GPUs:")
            for i in range(n_gpus):
                print(f" - {torch.cuda.get_device_name(device=i)}")
        else:
            if args.devices > n_gpus:
                print(f"Requested number of GPUs is superior to the number of GPUs available on this machine ({n_gpus}).")
                print(f"Setting number of used GPUs to maximum.")
                args.devices = n_gpus
            else:
                print(f"Using {args.devices} GPU(s):")
            for i in range(args.devices):
                print(f" - {torch.cuda.get_device_name(device=i)}")
    # Print the number of cores used if CPU is selected
    elif args.accelerator == "cpu":
        if args.devices == "auto":
            print(f"Using all {n_cpus} CPU cores.")
        else:
            if args.devices > n_cpus:
                print(f"Requested number of CPU cores is superior to the number of CPU cores available on this machine ({n_cpus}).")
                print("Setting number of used CPU cores to maximum.")
                args.devices = n_cpus
            print(f"Cores used: {args.devices}")

    if args.workers > n_cpus:
        print("Requested number of workers is superior to the number of CPU cores available on this machine." )
        print("Setting number of workers to maximum.")
        args.workers = n_cpus
    print(f"Number of workers used: {args.workers}")

    print(f"Maximum number of epochs: {args.epochs}")
    print(f"Batch size: {args.bs}")
    print(f"Initial learning rate: {args.lr}")
    print(f"Pretrained: {args.pretrained}")

    return args


