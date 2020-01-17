#### Env
python 3.7.3 \
numpy 1.16.2 \
pandas 0.24.2

#### Scripts:
##### data_loader.py : 
used to load data from json and convert to desired format
##### main_runner.py : 

main implementation

* func_timer: timer decorator
* func_dump: simplified dump output decorator
* limited_memoize : simplified memoize decorator
* TripMatch: main implementation with place holders for Fourier transform, Wavelet transform etc.
    * run_xcor : benchmark result execution
    
##### Preliminary Analysis and Result.html  
Jupyter method overview and result illustration(.html & ipynb)

##### run_xcor.pickle/run_xcor_new.pickle
Sample output from setting different thresholds :
run_xcor : setting mismatching time frame < 10s
run_xcor_new : setting mismatching time frame < 50s 

---------------------
**If pickle loader return package/import issue, please make sure to down grade package to the same version as listed or
run data_loader.py to regenerate data. 

