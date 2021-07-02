# utility functions for getting dset info
# 0. list of all terminal folders
# 1. list of files between 100-300 slices
# 2. metatadata df (size, spacing, direction) of all files
# 3. unique df (unique size, spacing, direction)
# 4. iso sizes


# Utilities
import os
import time
import pickle

# Input IO
import SimpleITK as sitk
import meshio

# Numpy and Pandas
import numpy as np
import torch
import pandas as pd
from pandas import DataFrame as DF

# modified os.walk to stop at last subdir
def get_terminal_folders(top):
    names = os.listdir(top)
    subdirs = [name for name in names if os.path.isdir(os.path.join(top, name))]

    mr_paths = []
    
    # terminal folder
    if len(subdirs) == 0:
        mr_paths.append(top)

    # recurse on subdirs
    for subdir in subdirs:
        newpath = os.path.join(top, subdir)
        mr_paths += get_terminal_folders(newpath)
    
    return mr_paths

# def get_terminal_folders(top):    
        
#     # start timer
#     start = time.time() 
    
#     # get series paths
#     mr_paths = walk_to_series(top)
        
#     # end timer
#     elapsed = time.time() - start
#     print(f"Elapsed: {elapsed} s for {len(mr_paths)} files.")
    
#     # save results
#     with open(f"{top}.txt", "wb") as fp:   #Pickling
#         pickle.dump(mr_paths, fp)
        
#     return mr_paths

# change path prefix
def change_src(s, old_src="../../../mnt/d/PitMRdata", new_src="/gpfs/data/oermannlab/private_data/DeepPit"): 
    return new_src + s[len(old_src):] 

# works for ABIDE
def get_folder_name(s):
    return int(search("\/\d{5}\/", s).group(0)[1:-1])

# pattern match file name for common sequences
def get_imputed_seq(fn):
    for seq in ("MPR", "RAGE", "T1", "T2", "FLAIR", "WOW", "GLOBAL"):
        if seq.lower() in fn or seq.upper() in fn:
            if seq == "RAGE": return "MPR"
            else: return seq
    return "UNKNOWN"

# returns extension, file w/ ext, and total number of files w/ ext
def get_ext(dir_path, exts=(".nii", ".dcm", ".img")):
    files    = os.listdir(dir_path)

    # assume only 1 type of MR ext in child folder
    for file in files:
        for ext in exts:
            if file.endswith(ext) and not file.startswith("._"):
                return ext, f"{dir_path}/{file}", len([f for f in files if f.endswith(ext)])
    # not known ext
    new_ext = file[file.rindex("."):]
    return new_ext, f"{dir_path}/{file}", len([f for f in files if f.endswith(new_ext)])

# returns metadata df, df of disqualified files
def read_metadata_files(dir_paths):
    d = []
    disqualified = []
    skipped = []
    
    for dir_path in dir_paths:

        # get file ext (nii, dcm, etc)
        ext, file, nfiles = get_ext(dir_path)
        seq               = get_imputed_seq(dir_path)
        
        if ext not in (".nii", ".dcm", ".img"):
            skipped.append((file, f"NOT RECOGNIZED EXTENSION: {ext}. Represents {nfiles} in dir."))
            continue
            
        # ASSUMES only 1 nii in folder

        if ext == ".nii" or ext == ".img":
            try:
                # read meta data
                reader = sitk.ImageFileReader()
                reader.SetImageIO("NiftiImageIO")
                reader.SetFileName(file)
                reader.ReadImageInformation()
            except Exception as e:
                print("ERROR. Skipped file below. Exception below.")
                print(file)
                print(e)
                skipped.append((file, e))
                continue

            # get num slices
            sz = reader.GetSize()
            n  = min(sz)

        elif ext == ".dcm":
            try:
                # read meta data
                reader = sitk.ImageFileReader()
                reader.SetFileName(file)
                reader.ReadImageInformation()
            except Exception as e:
                print("ERROR. Skipped file below. Exception below.")
                print(file)
                print(e)
                skipped.append((file, e))
                continue

            # add n_slices to size    
            sz = reader.GetSize()
            if sz[2] == 1:
                n = nfiles
                sz = (sz[0], sz[1], n)
            else:
                n = min(sz)

        else:
            print(f"Weird ext - {ext}.")

        # save
        if n >= 100 and n <= 300:
            d.append({
                "fn": dir_path,
                "imputedSeq": seq,
                "sz": sz,
                "px": sitk.GetPixelIDValueAsString(reader.GetPixelID()),
                "sp": tuple(round(x,2) for x in reader.GetSpacing()),
                "dir": tuple(int(round(x,1)) for x in reader.GetDirection())
            })
        else:
            disqualified.append({
                "fn": dir_path,
                "imputedSeq": seq,
                "sz": sz,
                "px": sitk.GetPixelIDValueAsString(reader.GetPixelID()),
                "sp": tuple(round(x,2) for x in reader.GetSpacing()),
                "dir": tuple(int(round(x,1)) for x in reader.GetDirection())
            })
        
    # save dataframe
    d = DF(d)
     
    return d, DF(disqualified), skipped

# prints unique in list
def print_unique(vals, sep="*" * 30):
    unique, idxs, cnts = np.unique(vals, return_index=True, return_inverse=False, return_counts=True, axis=0)
    print(sep)
    print("Num unique = ", len(unique))
    print("Unique: ", *unique, sep=" ")
    print("Counts: ", *cnts, sep = " ")
    print("Idxs: ", idxs, sep = " ")
    print(sep)

# returns unique df
def get_unique_df(full_metadata_df):
    unique_df = full_metadata_df.groupby(['sz', 'sp', 'dir'], as_index=False).agg({'fn': 'first'})
    unique_df['cnts'] = full_metadata_df.groupby(['sz', 'sp', 'dir']).size().values
    return unique_df

# returns the new size if made 2mm isotropic
def get_iso_sz(new_sp, orig_sz, orig_sp): return [int(round(osz*ospc/new_sp)) for osz,ospc in zip(orig_sz, orig_sp)]

def get_range(szs):
    # range of sizes
    mins = torch.min(torch.tensor(szs), dim=0)
    maxs = torch.max(torch.tensor(szs), dim=0)

    # print range
    for axis, minval, maxval in zip(("i", "j", "k"), mins.values, maxs.values):
        print(f"{axis}: {minval}-{maxval}")
        
    # return maxs
    return maxs.values

# returns (a) list of sizes and (b) max size
def get_iso_szs(df, new_sp):
    return [get_iso_sz(new_sp, orig_sz, orig_sp) for orig_sz, orig_sp in zip(list(df.sz.values), list(df.sp.values))]