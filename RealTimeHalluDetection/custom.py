import argparse
import logging
import os
import re
import random
import pandas as pd

import datasets
import torch
import torch.nn.functional as F

import ssl
import urllib.request
import zipfile

import time
import csv
import tqdm
import json

import warnings

import transformers
import accelerate
from accelerate import Accelerator
from accelerate import InitProcessGroupKwargs

from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    SchedulerType,
)
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
from transformers.generation.stopping_criteria import StoppingCriteriaList, LLamaQaStoppingCriteria

import numpy as np
from termcolor import colored
import datetime

"""
# TODO: Add the following functions

## Real-time hallucination detection

## Functions depends on Hallucination Types
### When refering to external knowledge

### When detecting hallucination only within the parametric knowledge
"""