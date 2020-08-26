#!/bin/bash

python DLNetworkForProteinAbundancePreparation.py --i1 data_sets_for_testing/E_MTAB_2836_fpkms.tsv --o1 data_sets_for_testing/tissue29_rna_test_sample_2.tsv --i2 data_sets_for_testing/E_PROT_29.tsv --o2 data_sets_for_testing/tissue29_prot_test_sample_2.tsv --id "GeneID" --cs --nthr "0.5" 1>data_sets_for_testing/normalization_messages_sample_2.txt 2>data_sets_for_testing/normalization_error_messages_sample_2.txt

python DLNetworkForProteinAbundancePrediction.py data_sets_for_testing/tissue29_prot_test_sample.tsv data_sets_for_testing/tissue29_rna_test_sample.tsv data_sets_for_testing/prediction_results_sample_2.txt data/gene_annotations_human.txt 1>data_sets_for_testing/prediction_messages_sample_2.txt 2>data_sets_for_testing/prediction_error_messages_sample_2.txt