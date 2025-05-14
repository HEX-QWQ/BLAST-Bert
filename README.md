# BLAST-Bert

This Project is based on [Bert-Pytorch](https://github.com/codertimo/BERT-pytorch/tree/master)

## Introduction

BLAST-Bert is a transformer-based language model designed to process and analyze biological sequence data by leveraging BLAST (Basic Local Alignment Search Tool) alignment results. Unlike traditional transformer models that rely on positional embeddings to encode token order, BLAST-Bert replaces positional embeddings with similarity scores derived from BLAST alignments. This adaptation enables the model to focus on the similarity relationships between a query sequence and database sequences, treating the database sequence accessions and their corresponding similarity scores as input "sentences." 




## Dataset

```
qseqid	sseqid	pident	length	mismatch	gapopen	qstart	qend	sstart	send	evalue	bitscore
Picornaviridae_Enterovirus_rhinovirus_A30_VMR1011691_51	picorna-like_Picornaviridae_YP_009505608.1_Human_rhinovirus_A1	80.9	2158	403	7	6	2154	1	2157	0.0	3634
```

## Input Specification 

1. ***Accession List:*** A list of database sequence accession IDs retrieved from a BLAST search, representing sequences aligned to the query.
2. ***Similarity Score List:*** A corresponding list of numerical similarity scores that quantify the alignment quality between the query sequence and each database sequence.

Formally, the input can be represented as:

Input Format: A tuple $(A, S)$, where:

- $A = [a_1, a_2, ..., a_n]$ is a list of (n) accession IDs (strings).
- $S = [s_1, s_2, ..., s_n]$ is a list of (n) similarity scores (floating-point numbers).

## Pre-Training Tasks

we design two pre-training tasks inspired by Bert, adapted to the context of biological sequence alignments:

- Masked Language Modeling (MLM): Randomly mask a subset of the accession IDs in the input accession list, and train the model to predict the masked accession IDs based on the remaining accessions and their associated similarity scores.

- Accession Coherence Predict (ACP): Train the model to determine whether a pair of accession IDs and their corresponding similarity scores originate from the same query sequence.