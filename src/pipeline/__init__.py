import sys, os, site
global_path = site.getsitepackages()[0]
sys.path.append(os.path.abspath(global_path+'/pipeline'))
#sys.path.append(os.path.abspath('./src/pipeline'))
"""
from pipeline.processing import frameproc
from pipeline import masking
from pipeline import sky
from pipeline import utils
from pipeline import photometry
"""