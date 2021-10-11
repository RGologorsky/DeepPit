import os

# download links
ppmi_full  = "https://www.dropbox.com/sh/bxuw198yx9ssvtl/AAAsZi9DWQFVefd3L7BtvwXfa?dl=0"
icmb_full  = "https://www.dropbox.com/sh/f61zv1q5kirzljn/AADJtefmT3BFZATi2doRUw8ka?dl=0"
adni1_full = "https://www.dropbox.com/sh/ibuuy9iixazg7vz/AADomHbTNxmf7sA72q6Up5i_a?dl=0"
aibl_full  = "https://www.dropbox.com/sh/1tq8gn8hnhpgzs0/AAAgoSLkoKnLyizhUtZ_JkGOa?dl=0"
abvib_full = "https://www.dropbox.com/sh/3it37rw92mgqw13/AABBJq8QMHvXwe6u25BNvt19a?dl=0"

zip_dir   = "/gpfs/data/oermannlab/private_data/DeepPit/PitMRdata"
unzip_dir = f"{zip_dir}/samir_labels"

# commands curl -L -o newName.zip https://www.dropbox.com/sh/[folderLink]?dl=1
names  = ["PPMI_full", "ICMB_full", "ADNI1_full", "AIBL_full", "ABVIB_full"]
links  = [ppmi_full, icmb_full, adni1_full, aibl_full, abvib_full]
curls  = [f"curl -L -o {zip_dir}/{name}.zip {link[:-1] + '1'}" for name, link in zip(names, links)]

# unzip -d samir_labels/PPMI_3107-3326 PPMI_3107-3326.zip
unzips = [f"unzip -d {unzip_dir}/{name} {zip_dir}/{name}.zip" for name in names]

print(curls[0])
print(unzips[0])

# execute curl
# for curl in curls:
#     result = os.popen(curl).read()
#     print(result)

# execute unzip
# for unzip in unzips:
#     result = os.popen(unzip).read()
#     print(result)