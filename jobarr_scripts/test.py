import argparse

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--time", default=10, type=int)
parser.add_argument("-n", "--nchunks", default=10, type=int)
parser.add_argument("-o", "--output", default="ABIDE", type=str)
parser.add_argument("-s", "--script", default="DeepPit/11b_save_N4_bias_all.py", type=str)
args = parser.parse_args()

time_str = f"{args.time:02}"  

#print(args.n)
print(args.nchunks)
