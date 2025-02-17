# Classification on S1SLC

For running the code, you need to install the library and provide a
configuration file (we provide a designated configuration file in the configs folder).

A path to the dataset must be provided in the configuration file: we suggest to specify the directory of the unzipped S1SLC dataset.

To download the S1SLC dataset, please follow the following link: https://ieee-dataport.org/open-access/s1slccvdl-complex-valued-annotated-single-look-complex-sentinel-1-sar-dataset-complex. Warning: you must be logged to download the dataset.

Finally, either by using the provided config_s1slc.yml or a configuration file of your own (be careful to correctly set name:S1SLC), you can start the training of your model:

```bash
python -m torchtmpl.main train configs/config_s1slc.yml 

```

To retrain a model:

```bash
python -m torchtmpl.main retrain logs/path_to_model 

```

To test a model (and get its performances metrics):

```bash
python -m torchtmpl.main test logs/path_to_model 

```

If you are using the submit-slurm.py script to run the training, make sure to set the number of retrains to 4 (to achieve at least 100 epochs).