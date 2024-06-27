from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import trange
from tqdm.autonotebook import tqdm

from data_loader.hybrid_data_loaders import *
from data_loader.header_data_loaders import *
from data_loader.CT_Wiki_data_loaders import *
from data_loader.RE_data_loaders import *
from data_loader.EL_data_loaders import *
from model.configuration import TableConfig
from model.model import HybridTableMaskedLM, HybridTableCER, TableHeaderRanking, HybridTableCT,HybridTableEL,HybridTableRE,BertRE
from model.transformers import BertConfig,BertTokenizer, WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup
from utils.util import *
from baselines.row_population.metric import average_precision,ndcg_at_k
from baselines.cell_filling.cell_filling import *
from model import metric


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'CER': (TableConfig, HybridTableCER, BertTokenizer),
    'CF' : (TableConfig, HybridTableMaskedLM, BertTokenizer),
    'HR': (TableConfig, TableHeaderRanking, BertTokenizer),
    'CT': (TableConfig, HybridTableCT, BertTokenizer),
    'EL': (TableConfig, HybridTableEL, BertTokenizer),
    'RE': (TableConfig, HybridTableRE, BertTokenizer),
    'REBERT': (BertConfig, BertRE, BertTokenizer)
}


data_dir = r"G:\CPSC448\TURL\data\wikitables_v2"

config_name = "configs/table-base-config_v2.json"
device = torch.device('cuda')
# load entity vocab from entity_vocab.txt
entity_vocab = load_entity_vocab(data_dir, ignore_bad_title=True, min_ent_count=2)
entity_wikid2id = {entity_vocab[x]['wiki_id']:x for x in entity_vocab}

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

with open(os.path.join(data_dir, 'test.all_table_col_type.json'), 'r') as f:
    table1 = json.load(f)[0]

with open(os.path.join(data_dir, 'test.table_col_type.json'), 'w') as json_file:
    json.dump([table1], json_file)

# load type vocab from type_vocab.txt
type_vocab = load_type_vocab(data_dir)
test_dataset = WikiCTDataset(data_dir, entity_vocab, type_vocab, max_input_tok=500, src="test", max_length = [50, 10, 10], force_new=False, tokenizer = None)