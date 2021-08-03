# Intent Classification and Slot Filling for Privacy Policies

Official code release of our work on [Intent Classification and Slot Filling for Privacy Policies](https://aclanthology.org/2021.acl-long.340/). 

- We propose a new dataset called PolicyIE in this work. 
- PolicyIE provides annotations of privacy practices and text spans for sentences in policy documents. 
- We refer to predicting privacy practice as *intent classification* and identifying the text spans as *slot filling*. 

**[NOTE]** The PolicyIE dataset is available [here](https://github.com/wasiahmad/PolicyIE/blob/main/data/sanitized_split.zip).


### Dependencies
- python>=3.6
- torch==1.5.1
- transformers==3.0.2
- fairseq==0.9.0
- seqeval==1.2.0
- pytorch-crf==0.7.2


### Data Preparation

```bash
cd data
bash prepare.sh
```


### Models

We studied the following two alternative modeling approaches as baselines in our work. We refer the readers to the 
paper for more details about the models and experiment results.


#### Joint Intent and Slot Tagging

```
# Input
[CLS] We may also use or display your username and icon or profile photo on marketing purpose or press releases .

# Type-I slot tagging output
Data-Collection-Usage B-DC.FPE O O B-Action O O B-DP.U B-DC.UOAP O B-DC.UOAP I-DC.UOAP I-DC.UOAP I-DC.UOAP O O O O O O O

# Type-II slot tagging output
Data-Collection-Usage O O O O O O O O O O O O O O B-P.AM I-P.AM I-P.AM I-P.AM I-P.AM O
```

- **[Models]** BiLSTM, Transformer, [BERT](https://arxiv.org/abs/1810.04805), [RoBERTa](https://arxiv.org/abs/1907.11692)
- Implementations are available at https://github.com/wasiahmad/PolicyIE/tree/main/seqtag.
- Go to the `seqtag` directory and use the `run.sh` script for model training and evaluation. Run `bash run.sh -h` to 
learn about the command line arguments.


#### Sequence-to-sequence Learning

```
# Input
We may also use or display your username and icon or profile photo on marketing purpose or press releases .

# Output
[IN:Data-Collection-Usage [SL:DC.FPE We] [SL:Action use] [SL:DP.U your] [SL:DC.UOAP username] [SL:DC.UOAP icon or profile photo] [SL:P.AM marketing purpose or press releases]]
```

- **[Models]** [UniLM](https://arxiv.org/pdf/1905.03197.pdf), [UniLMv2](https://arxiv.org/pdf/2002.12804.pdf), [MiniLM](https://arxiv.org/pdf/2002.10957.pdf), [BART](https://arxiv.org/pdf/1910.13461.pdf)
- Implementations are available at https://github.com/wasiahmad/PolicyIE/tree/main/{bart,mass,unilm}.
- Go to the corresponding model directory and use the `prepare.sh` script to prepare data and `run.sh` script for 
model training and evaluation. Run `bash run.sh -h` to learn about the command line arguments.


#### Acknowledgement

We acknowledge the efforts of the authors of the following repositories.

- https://github.com/monologg/JointBERT
- https://github.com/microsoft/unilm
- https://github.com/microsoft/MASS
- https://github.com/pytorch/fairseq/tree/master/examples/bart


#### Citation

```
@inproceedings{ahmad-etal-2021-intent,
    title = "Intent Classification and Slot Filling for Privacy Policies",
    author = "Ahmad, Wasi  and
      Chi, Jianfeng  and
      Le, Tu  and
      Norton, Thomas  and
      Tian, Yuan  and
      Chang, Kai-Wei",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.340",
    doi = "10.18653/v1/2021.acl-long.340",
    pages = "4402--4417",
}
```
