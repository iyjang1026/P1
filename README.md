# astronomical_pipeline
completed python based astronomical pipeline

# Background set

conda
---
create env
```bash
conda create -n astro python=3.12 --platform osx-64 # for mac

conda create -n astro python=3.12 --platform linux-64 # for linux
```

conda-forge set
```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
```

install software
```bash
conda install conda-forge::astromatic-<software> # for astromatic softwares, like swarp, scamp, source-extractor, psfex

conda install conda-forge::astrometry # astrometry.net
```

python pkg
---
astropy, astroquery, photutils, scipy, scikit-image, ray(for multi-processing) etc. If you install P1 pipeline package, then these packges are installed with P1 pipeline package.

Install P1 package
---
1. install or git clone this repository
2. cd installed directory and type terminal this,
```bash 
pip install -e .
```

Initial folder setting
---
Select location And make folder with name you want. The folder must have named folder that is "LIGHT, DARK, BIAS." The names are must be writen by Capital Alphabet.

photometric catalog must be "sdss" and named "sdss_<target_name>.csv"
