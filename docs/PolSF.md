# Semantic segmentation on PolSF

For running the code, you need to install the library and provide a
configuration file (we provide a designated configuration file in the configs folder).

A path to the dataset must be provided in the configuration file: we suggest to specify the directory of the unziped PolSF dataset.

To download the PolSF dataset, please follow the following link: https://ietr-lab.univ-rennes1.fr/polsarpro-bio/san-francisco/dataset/SAN_FRANCISCO_ALOS2.zip. Warning: you must move the file SF-ALOS2-label2d.png from the sub-folder Config_labels to the main folder.

Finally, either by using the provided config_polsf.yml or a configuration file of your own (be careful to correctly set name:PolSFDataset), you can start the training of your model:

```bash
python -m torchtmpl.main train configs/config_polsf.yml 

```

To retrain a model:

```bash
python -m torchtmpl.main retrain logs/path_to_model 

```

To test a model (and get its performances metrics):

```bash
python -m torchtmpl.main test logs/path_to_model 

```