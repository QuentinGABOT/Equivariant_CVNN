# Equivariant_CVNN

For running the code, you need to install the library and provide a
configuration file. 

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install .
```

Then, you specify the experiment you want to run with a `config.yaml` file.
Check the example `config.yaml` we provide. Then :

```bash
python -m torchtmpl.main config.yml train 

```
