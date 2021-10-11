gpu=true

for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)   

    case "$KEY" in
            gpu)                gpu=${VALUE} ;;
            REPOSITORY_NAME)    REPOSITORY_NAME=${VALUE} ;;     
            *)   
    esac    
done

if [ $gpu = "true"  ]; then
    srun -p gpu4_dev --gres=gpu:1 --time=00-03:00:00 --cpus-per-task=5 --pty bash
else
    srun -p cpu_dev --mem-per-cpu=8G --time=00-02:00:00 --cpus-per-task=5 --pty bash
fi

