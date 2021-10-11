import os, sys, subprocess

models    = ["UNET3D", "VNET", "OBELISKHYBRID", "CONDSEG"]

models    = ["UNET3D" for _ in range(8)]
models   += ["UNETR"  for _ in range(4)]

print(models)

#models    = ["UNETR", "CONDSEG", "OBELISKHYBRID", "VNET", "UNET3D"]

loss_fns  = ["DICE_loss", "BCE_loss"]

do_simple_bools  = [True]
#do_simple_bools = [True, False]

# full res pixdim for OBELISK
cmds = [f"python singbatch.py --model_type {model_type} --loss_type {loss_type} --pixdim 1.5 --full_res 96 --do_simple {do_simple}"
        for model_type in models
        for loss_type in loss_fns
        for do_simple in do_simple_bools
       ]

# add OBELISK 144
cmds += [f"python singbatch.py --model_type OBELISKHYBRID --loss_type {loss_type} --pixdim 1.0 --full_res 144 --do_simple {do_simple}"
        for loss_type in loss_fns
        for do_simple in do_simple_bools
       ]

# repeat -- 20 models (3 + OBELISK x2, 2 loss x 2 do_simple)
print(f"len cmds {len(cmds)}", *cmds, sep="\n")

# execute
for cmd in cmds:
    os.system(cmd)  



#   999  python singbatch.py --model_type UNET3D --loss_type DICE_loss --pixdim 1.5 --full_res 96 --do_simple True
#  1000  python singbatch.py --model_type UNET3D --loss_type BCE_loss --pixdim 1.5 --full_res 96 --do_simple True
#  1001  python singbatch.py --model_type VNET --loss_type BCE_loss --pixdim 1.5 --full_res 96 --do_simple True
#  1002  python singbatch.py --model_type VNET --loss_type DICE_loss --pixdim 1.5 --full_res 96 --do_simple True
#  1003  python singbatch.py --model_type CONDSEG --loss_type DICE_loss --pixdim 1.5 --full_res 96 --do_simple True
#  1004  python singbatch.py --model_type CONDSEG --loss_type BCE_loss --pixdim 1.5 --full_res 96 --do_simple True
#  1005  python singbatch.py --model_type OBELISKHYBRID --loss_type BCE_loss --pixdim 1.5 --full_res 96 --do_simple True
#  1006  python singbatch.py --model_type OBELISKHYBRID --loss_type DICE_loss --pixdim 1.5 --full_res 96 --do_simple True
#  1007  python singbatch.py --model_type OBELISKHYBRID --loss_type DICE_loss --pixdim 1.0 --full_res 144 --do_simple True
#  1008  python singbatch.py --model_type OBELISKHYBRID --loss_type BCE_loss --pixdim 1.0 --full_res 144 --do_simple True
