import time

# time_one_batch
# get_time_id

def time_one_batch(dls):
    # time it
    start = time.time()

    b = dls.one_batch()
    print(f"Batch is type {type(b)}, x: {b[0].shape}, y: {b[1].shape}")
    print(f"Train ds: {len(dls.train_ds)}, Valid ds: {len(dls.valid_ds)}")
    print(f"Train dls #batches: {len(dls.train)}, Valid dls #batches: {len(dls.valid)}")

    # end timer
    elapsed = time.time() - start
    print(f"Elapsed time: {elapsed} s.")
    
    return b
    
def get_time_id():
    return f'''{int(time.time())}_{time.strftime("%a_%b_%d_%Y_hr_%H_min_%M")}'''