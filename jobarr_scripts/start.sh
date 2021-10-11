load=true
bind=true
jup=false
scriptname=false
train=false
shel=false
sif=new_monai_1906

convert=true
model_type=UNET3D
loss_type=TEST
pixdim=1.5
full_res=96
do_flip=true

for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   

    case "$KEY" in
            load)              load=${VALUE} ;;
            bind)              bind=${VALUE} ;;
            jup)                jup=${VALUE} ;;     
            scriptname)  scriptname=${VALUE} ;;
            train)            train=${VALUE} ;;
            shel)              shel=${VALUE} ;;
             sif)               sif=${VALUE} ;;
      model_type)        model_type=${VALUE} ;;
       loss_type)         loss_type=${VALUE} ;;
          pixdim)            pixdim=${VALUE} ;;
        full_res)          full_res=${VALUE} ;;
         do_flip)           do_flip=${VALUE} ;;
            *)   
    esac    
done

if [ $load = "true" ]; then
 module purge; module load python; module load singularity;
fi

if [ $jup = "true" ]; then
  singularity exec --bind /gpfs/data/oermannlab/private_data/DeepPit/:/gpfs/data/oermannlab/private_data/DeepPit/ --nv "$sif.sif" jupyter notebook --ip=0.0.0.0 --port=8889
fi

if [ $convert = "true" ]; then
  singularity exec "$sif.sif" bash -c "jupyter nbconvert --to script $scriptname.ipynb" 
fi

if [ $train = "true" ]; then
  singularity exec --bind /gpfs/data/oermannlab/private_data/DeepPit/:/gpfs/data/oermannlab/private_data/DeepPit/ --bind /gpfs/home/gologr01/DeepPit/:/DeepPit/ --nv "$sif.sif" python3 DeepPit/mylaunch.py $scriptname.py model_type=$model_type loss_type=$loss_type pixdim=$pixdim full_res=$full_res do_flip=$do_flip
fi

if [ $shel = "true" ]; then    
    singularity shell --bind /gpfs/data/oermannlab/private_data/DeepPit/:/gpfs/data/oermannlab/private_data/DeepPit/ --nv "$sif.sif"
fi
