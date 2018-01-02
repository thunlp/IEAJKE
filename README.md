# IEAJKE
Codes and data for our paper "[Iterative Entity Alignment via Joint Knowledge Embeddings](https://www.ijcai.org/proceedings/2017/0595.pdf)"

The code for ITransE and DFB-1 are available now!

## Dataset
For DFB-1, we provide four files
* common_entities.txt: alignment seeds;
* common_entities2id.txt: id version of alignment seeds;
* newentity2id.txt: the correspondence of entities and their ids;
* relation2id.txt: the correspondence of relations and their ids;
* triple2id.txt: all triples in FB15K. Note that this is not the original form, but a processed one where entity seeds are already combined as one.

## How to run
First, compile src/transE.cpp by:
`g++ transE.cpp -o transE -pthread -O3 -std=c++11 -march=native`

The compilation arguments maybe different from platform to platform. Such compilation command is the same as the one in [Fast-TransX](https://github.com/thunlp/Fast-TransX). If you come across problems, you can refer to the issues of [Fast-TransX](https://github.com/thunlp/Fast-TransX).

Run `./transE`.

To test the result, run test.ipynb by Jupyter Notebook. We also provide pretrained embeddings in entity2vec.bern. You should get "0.8309717616319968 0.6708873480052256 80.05868756908853 9951" finally in the sixth block.

# Contact
If you have any problems about my paper and datsets, please send email to zhuhao15@mails.tsinghua.edu.cn.
