# This script predicts  missing protein values from rna values and optional gene context file.
# usage: DLNetworkForProteinAbundancePrediction.py rna_file prot_file out_file [context_file]
# missing protein values should be marked with 0 or NA in the input protein file
import numpy as np
import os
import tensorflow as tf
import csv
import random
from utils.protein_predict_model_a import protein_model
import time
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '20'
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

use_gene_ids = False # should not be used together with testset_by_genes
use_rna = True # use of rna value for prediction

batchSize=32
display_step = 5000
training_iters = 60001

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

class data_processor:
    def __init__(self, seed):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        self.dataset = [] #[dims, type, rna_val, prot_val, p_length, gene_id, rp_data]
        self.n_genes = 1
        self.cur_ind = 0
        self.onto_count = 0

    def read_onto(self, onto_name):
        self.onto_set = {}
        if onto_name is not None:
            with open(onto_name) as csvfile:
                onto_reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
                header_row = next(onto_reader)
                self.onto_count = len(header_row) - 1
                assert self.onto_count >= 0
                names_onto = [set() for i in range(self.onto_count)]

                for row in onto_reader:
                    gene_id = row[0]
                    dims = []
                    for column in range(self.onto_count):
                        onto_ids = row[1+column].split(',') if len(row[1+column])>0 else []
                        names_onto[column] |= set(onto_ids)
                        dims.append(onto_ids)
                    self.onto_set[gene_id]=dims

            self.dimensions_len = [len(names_onto[k]) for k in range(self.onto_count)]
            print("onto items:", self.dimensions_len)
            self.c_dict = []
            for k in range(self.onto_count):
                dd = {x: id for id, x in enumerate(names_onto[k])}
                self.c_dict.append(dd)

            self.dimensions = sum(self.dimensions_len)
            print("dimensions = ", self.dimensions)
        else:
            self.dimensions = 0
        return

    def readData(self, rna_name, prot_name, onto_name=None):
        self.read_onto(onto_name)

        with open(rna_name) as csvfile, open(prot_name) as protfile:
            rna_reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
            prot_reader = csv.reader(protfile, delimiter='\t', quotechar='"')
            nr = 0
            for row in rna_reader:
                protrow = next(prot_reader)
                if nr > 0:  # skip header line
                    if row[0] != protrow[0]: raise("not equal prot and rna ids")
                    if use_gene_ids: gene_id = row[0]
                    else: gene_id = 0
                    dims = []
                    dim_item = self.onto_set[row[0]] if row[0] in self.onto_set else [[]]*self.onto_count
                    for dim in range(self.onto_count):
                        arrC = np.zeros([self.dimensions_len[dim]], dtype=np.float32)
                        for x in dim_item[dim]: arrC[self.c_dict[dim][x]] = 1.0
                        dims.append(arrC)

                    if len(dims) > 0:dims = np.concatenate(dims)
                    rna_row = []
                    prot_row= []
                    for type in range(1, len(row)): # the first item in row is gene id
                        rna_txt = row[type]
                        prot_txt = protrow[type]
                        prot_val = 0.0
                        rna_val = 0.0
                        if is_number(rna_txt):
                            rna_val = np.log(float(rna_txt) + 1) if use_rna else 0
                        if is_number(prot_txt) and float(prot_txt) > 0:  # keep only nonzero entries
                            prot_val = np.log(float(prot_txt) + 1)

                        rna_row.append(rna_val)
                        prot_row.append(prot_val)

                    for type in range(1, len(row)): # the first item in row is gene id
                        rna_txt = row[type]
                        prot_txt = protrow[type]
                        prot_val = prot_row[type-1]
                        rna_val = rna_row[type-1]
                        other_rna = 0.0
                        rp_data = []
                        self.rp_length = len(rp_data)
                        item = [dims, type - 1, rna_val, prot_val, other_rna, gene_id, rp_data]

                        if is_number(rna_txt) and is_number(prot_txt):
                            if float(prot_txt) > 0:  # keep only nonzero entries
                                self.dataset.append(item)

                nr+=1

    # predict missing values
    def predict(self,rna_name, prot_name, onto_name, model, sess, out_prot_fname):
        print("Predicting...")
        with open(rna_name) as csvfile, open(prot_name) as protfile, open(out_prot_fname, 'w') as outfile:
            rna_reader = csv.reader(csvfile, delimiter='\t', quotechar='"')
            prot_reader = csv.reader(protfile, delimiter='\t', quotechar='"')
            resultwriter = csv.writer(outfile, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            rna_header = next(rna_reader)
            prot_header = next(prot_reader)
            resultwriter.writerow(prot_header)

            for row in rna_reader:
                protrow_txt = next(prot_reader)
                if row[0] != protrow_txt[0]: raise ("not equal prot and rna ids")
                if use_gene_ids:
                    gene_id = row[0]
                else:
                    gene_id = 0
                dims = []
                dim_item = self.onto_set[row[0]] if row[0] in self.onto_set else [[]] * self.onto_count
                for dim in range(self.onto_count):
                    arrC = np.zeros([self.dimensions_len[dim]], dtype=np.float32)
                    for x in dim_item[dim]: arrC[self.c_dict[dim][x]] = 1.0
                    dims.append(arrC)

                if len(dims) > 0: dims = np.concatenate(dims)
                rna_row = []
                prot_row = []
                for type in range(1, len(row)): # the first item in row is gene id
                    rna_txt = row[type]
                    prot_txt = protrow_txt[type]
                    prot_val = 0.0
                    rna_val = 0.0
                    if is_number(rna_txt):
                        rna_val = np.log(float(rna_txt) + 1) if use_rna else 0
                    if is_number(prot_txt) and float(prot_txt) > 0:  # keep only nonzero entries
                        prot_val = np.log(float(prot_txt) + 1)

                    rna_row.append(rna_val)
                    prot_row.append(prot_val)

                for type in range(1, len(row)): # the first item in row is gene id
                    rna_txt = row[type]
                    prot_txt = protrow_txt[type]
                    rna_val = rna_row[type-1]
                    other_rna = 0.0
                    rp_data = []

                    if is_number(rna_txt):
                        if not is_number(prot_txt) or float(prot_txt)<=0:
                            gene_id_nr = 0
                            if gene_id in self.object_dict:
                                gene_id_nr = self.object_dict[gene_id]
                            elif use_gene_ids:
                                raise RuntimeError("Fail: using gene ids, but for gene "+gene_id+" not a single protein value exists")
                            result = model.getResult(sess, [dims], [type-1], [rna_val], [other_rna], [gene_id_nr], [rp_data])
                            result_value = result[0, 0]
                            protrow_txt[type] = np.exp(result_value)-1
                resultwriter.writerow(protrow_txt)

    def prepare(self, rna_name, prot_name, onto_name=None):
        self.readData(rna_name, prot_name, onto_name)
        self.preprocess()

    # convert gene ids to integers
    def preprocess(self):
        d, t, x, y, l, gene_id, rp_data = zip(*self.dataset)
        id_list = list(gene_id)
        id_set = set(id_list)
        self.object_dict = {x:nr for x, nr in zip(id_set, range(len(id_set)))}
        int_list = [self.object_dict[x] for x in id_list]
        self.dataset = list(zip(d, t, x, y, l, int_list, rp_data))

        gene_ids = list(zip(*self.dataset))[5]
        self.n_genes = max(gene_ids)+1
        #print("number of genes = ", self.n_genes)
        type_ids = list(zip(*self.dataset))[1]
        self.dimensionsType = max(type_ids) + 1
        print("tissue types = ", self.dimensionsType)
        self.train_set = self.dataset
        print("train set length:", len(self.train_set))

    def get_batch(self):
        batch = []
        for itemNr in range(batchSize):
            elem = self.train_set[self.cur_ind]
            self.cur_ind += 1
            if self.cur_ind >= len(self.train_set):
                random.shuffle(self.train_set)
                self.cur_ind = 0
            batch.append(elem)
        return batch

    def train(self):
        tf.set_random_seed(self.seed)        #disable weight randomization
        model = protein_model(self.dimensions, self.dimensionsType, self.n_genes if use_gene_ids else 0, self.rp_length)
        model.createTrainGraph()
        model.createResultGraph()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        step=1
        sumLoss = 0.0
        start_time = time.time()
        print("Computing...")
        while step < training_iters:
            if step % display_step == 0:
                step_time = time.time() - start_time
                start_time = time.time()
                print ("step = ", step, "loss = ", sumLoss / display_step, "time=", step_time)
                sumLoss = 0.0

            batch = self.get_batch()
            d,t,x,y,l, genes, rp_data = zip(*batch)
            loss = model.train(sess, d,t,x,y,l, genes, rp_data)
            sumLoss+=loss
            step+=1

        return model, sess

argv = sys.argv[1:]
if(len(argv)<3): print("usage: DLNetworkForProteinAbundancePrediction.py rna_file prot_file out_file [context_file]")

# rna_fname = 'data/nci60_final_rna_without_na.txt'
# prot_fname = 'data/nci60_final_proteomics_without_na.txt'
# out_prot_fname = 'out_prot.csv'
onto_fname = None#'data/GeneContext50.csv'

rna_fname = argv[0]
prot_fname = argv[1]
out_prot_fname = argv[2]
if len(argv)>=4: onto_fname = argv[3]

trainer = data_processor(50)
trainer.prepare(rna_fname, prot_fname, onto_fname)

model, sess = trainer.train()
trainer.predict(rna_fname, prot_fname, onto_fname, model, sess, out_prot_fname)
print('Done')