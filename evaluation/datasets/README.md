## Datasets

## Information

Disclaimer/Credit: This table was adapted from the [DEG Readme](https://github.com/Visual-Computing/DynamicExplorationGraph/blob/main/readme.md) and therefore also contains download-links to their own servers.   
It was, however, extended by the SIFTSMALL, the GIST1M and the LAION5B dataset.  
The latter was not anymore publicly downloadable but supplied by the supervisor of this work. The 3M subset and its ground_truth was manually derived from the 10M dataset. Note, that 1M subset groundtruth/gold-standards exist for download on the official website.

| Data set | Download | d | nd | nq | top_k | LID | File-Format |
|-|-|-|-|-|-|-|-|
| [Audio](https://www.cs.princeton.edu/cass/)| [audio.tar.gz](https://static.visual-computing.com/paper/DEG/audio.tar.gz) | 192 | 53,387 | 200 | 100 | 5.6 | fvecs |
| [DEEP1M](https://ieeexplore.ieee.org/document/7780595)  | [deep1m.tar.gz](https://static.visual-computing.com/paper/DEG/deep1m.tar.gz) | 96 | 1,000,000 | 10,000 | 100 | 15.5 | fvecs |
| [Enron](https://www.cs.cmu.edu/~enron/) | [enron.tar.gz](https://static.visual-computing.com/paper/DEG/enron.tar.gz) | 1369 | 94,987 | 200 | 20 | 11.7 | fvecs |
| [GIST1M](http://corpus-texmex.irisa.fr/) | [gist.tar.gz](ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz) | 960 | 1,000,000 | 1,000 | 100 | / | fvecs |
| [GloVe-100](https://nlp.stanford.edu/projects/glove/) | [glove-100.tar.gz](https://static.visual-computing.com/paper/DEG/glove-100.tar.gz) | 100 | 1,183,514 | 10,000 | 100 | 21.7 | fvecs |
| [LAION5B](https://sisap-challenges.github.io/2024/datasets/) | provided via supervisor | 768 | 300K, 3M, 10M | 10,000 | 1000 | / | HDF5 |
| [SIFT1M](http://corpus-texmex.irisa.fr/) | [sift.tar.gz](https://static.visual-computing.com/paper/DEG/sift.tar.gz) | 128 | 1,000,000 | 10,000 | 100 | 9.2 | fvecs |
| [SIFTSMALL](http://corpus-texmex.irisa.fr/) | [siftsmall.tar.gz](ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz) | 128 | 10,000 | 100 | 100| / | fvecs |


- Dataset downloading, e.g.,
    - `curl -o gist.tar.gz ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz`
    - `tar --exclude="gist_base.fvecs" --exclude="gist_learn.fvecs" -xvf gist.tar.gz`
