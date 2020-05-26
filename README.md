# MASTER'S THESIS
Master's Thesis Git Repository.

## Readme 

*Real readme coming soon.*

For now, find the Bayesian VAE and WaveNet reimplementations (in PyTorch) under ```src/models```. Each model has a python script in ```src/``` for running training instances with that model. For example
```
python run_vae.py --log_interval batch
```
trains a newly initialized VAE model on default arguments (which can be viewed using the ```--help``` or ```-h``` flag), with logging information per batch update (basically a training progress bar).

### Data
Place aligned data files under ```src/data/files/alignments/```. The alignment data used by default is from the [DeepSequence repository](https://github.com/debbiemarkslab/DeepSequence), which can easily be downloaded and extracted using their script, as provided here as well. Navigate to ```src/data/files/``` and run this script to download and extract alignment files:
```
curl -o alignments.tar.gz https://marks.hms.harvard.edu/deepsequence/alignments.tar.gz
tar -xvf alignments.tar.gz
rm alignments.tar.gz
```

