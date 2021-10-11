import os, pickle, glob
from pathlib import Path
"""
export 
(1) paths to code, data, models
(2) items - abide_lbl_items, cross_lbl_items, all_lbl_items
(3) getd (train & valid transforms in transforms.py)
"""

############## PATHS #####################################
# Paths to (1) code (2) data
code_src    = "/gpfs/home/gologr01"
data_src    = "/gpfs/data/oermannlab/private_data/DeepPit"

# stored code
deepPit_src = f"{code_src}/DeepPit"
obelisk_src = f"{code_src}/OBELISK"

# stored data
model_src   = f"{data_src}/saved_models"
label_src   = f"{data_src}/PitMRdata/samir_labels"
ABIDE_src   = f"{data_src}/PitMRdata/ABIDE"

# saved metadata
dsetmd_src   = f"{data_src}/saved_dset_metadata"

# stored runs Tensorboard
run_src     = f"{data_src}/runs"

############## LABEL ITEMS #####################################
cross_lbl_folders = ["PPMI_full", "ICMB_full", "ADNI1_full", "AIBL_full", "ABVIB_full"]
cross_lbl_folders = [f"{label_src}/{folder}" for folder in cross_lbl_folders]

abide_lbl_folders = ["50373-50453", "50313-50372", "50213-50312", "50155-50212", "50002-50153"]
abide_lbl_folders = [f"{label_src}/{folder}" for folder in abide_lbl_folders]

def get_las_n4(file):
    nii_paths   = glob.glob(f"{file}/*.nii")         
    las_n4_niis = [nii for nii in nii_paths if "las_n4" in nii or "las_corrected_n4" in nii]
    assert(len(las_n4_niis) == 1)
    return las_n4_niis[0]

# for abide
# make a dictionary of key = train folder, value = (segm obj, nii file)         
def get_data_dict_las_n4(train_path):
    train_folders   = os.listdir(train_path)
    train_data_dict = {}
    for folder in train_folders:
        segm_obj_path = os.path.join(train_path, folder, "seg.pt")

        mp_path      = os.path.join(train_path, folder, "MP-RAGE")
        folder1_path = os.path.join(mp_path, os.listdir(mp_path)[0])
        folder2_path = os.path.join(folder1_path, os.listdir(folder1_path)[0])

        # choose corrected_n4 if available
        nii_path = get_las_n4(folder2_path)
        train_data_dict[folder] = (nii_path, segm_obj_path) #(segm_obj_path, nii_path)
    return train_data_dict

# Get ABIDE data dict
abide_data = {}
for folder in abide_lbl_folders: abide_data.update(get_data_dict_las_n4(folder))

# Convert data dict => items (path to MR, path to Segm tensor)
abide_lbl_items = list(abide_data.values())
print(f"Full lbl items: {len(abide_lbl_items)}")

# remove bad label 50132
abide_weird_lbls = [50132, 50403]
def is_weird(fn): return any([str(lbl) in fn for lbl in abide_weird_lbls])
   
abide_lbl_items = [o for o in abide_lbl_items if not is_weird(o[0])]
print(f"Removed {len(abide_weird_lbls)} weird, new total lbl items: {len(abide_lbl_items)}")

# train/valid/test
with open(f"{data_src}/saved_dset_metadata/split_train_valid_test.pkl", 'rb') as f:
    train_idxs, valid_idxs, test_idxs, train_items, valid_items, test_items = pickle.load(f)
    tr_len, va_len, te_len =  len(train_items), len(valid_items), len(test_items)
    print("train, valid, test", tr_len, va_len, te_len, "total", tr_len + va_len + te_len)
    
# train_items = [items[i] for i in train_idxs]
# valid_items = [items[i] for i in valid_idxs]
# test_items  = [items[i] for i in test_idxs]

# Open matched segs = (seg_folder, mr_folder)
with open(f"{dsetmd_src}/first_418_matched_segs.txt", "r") as f:
    lst = f.read().splitlines()
    # str to tuple
    cross_mr_paths, cross_seg_paths = zip(*[eval(s) for s in lst])
    
# Get cross-dset data dict
cross_lbl_items = [(get_las_n4(mr), f"{seg}/seg.pt") for mr, seg in zip(cross_mr_paths, cross_seg_paths)]

all_lbl_items      = abide_lbl_items + cross_lbl_items
all_test_lbl_items = test_items + cross_lbl_items
print(f"Cross label items: ", len(cross_lbl_items))
print(f"All label items: ", len(all_lbl_items), f"(abide ({len(abide_lbl_items)}) + cross_lbl ({len(cross_lbl_items)}))")
print(f"Test label items: ", len(all_test_lbl_items), f"(test ({len(test_items)}) + cross_lbl ({len(cross_lbl_items)}))")

def getd(items): return [{"image": mr_path, "label": f"{Path(mk_path).parent}/seg.nii"} for mr_path, mk_path in items]