import bitsandbytes as bnb

  
# --- Main Execution ---
if __name__ == "__main__":
    print(bnb.__version__)
    print(bnb.cuda_setup.get_compute_capability())
