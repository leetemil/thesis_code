## Readme 

*~~Real readme coming soon.~~ 2020/05/26 snapshot of code.*

For now, find the Bayesian VAE and WaveNet reimplementations (in PyTorch) under ```src/models```. Each model has a python script in ```src/``` for running training instances with that model. For example
```
python run_vae.py --log_interval batch
```
from the ```src``` folder trains a newly initialized VAE model on default arguments (which can be viewed using the ```--help``` or ```-h``` flag), with logging information per batch update (basically a training progress bar). We currently get the best scores on VAE training 5 models, each with sparse interactions (```dict``` flag) and randomly weighted samples.
```
python run_vae.py -dict -rws
```
Then, use the ```evaluate_ensemble.py``` script in ```src``` to make ensemble predictions over the trained models. This script takes one or more directories as argument and looks for model files (all files with ```.torch``` suffix) to include in the ensemble.

### Data
Place aligned data files under ```src/data/files/alignments/```. The alignment data used by default is from the [DeepSequence repository](https://github.com/debbiemarkslab/DeepSequence), which can easily be downloaded and extracted using their script, as provided here as well. Navigate to ```src/data/files/``` and run this script to download and extract alignment files:
```
curl -o alignments.tar.gz https://marks.hms.harvard.edu/deepsequence/alignments.tar.gz
tar -xvf alignments.tar.gz
rm alignments.tar.gz
```

