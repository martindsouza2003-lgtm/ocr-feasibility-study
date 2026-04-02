

<div align="center">

<h3 align="center">PLATTER: A Page-Level Handwritten Text Recognition System for Indic Scripts</h3>

<p>  

[![arXiv](https://img.shields.io/badge/arXiv-2503.11932-b31b1b.svg)](https://arxiv.org/abs/2502.06172) &nbsp;
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97-Data-yellow)](https://huggingface.co/datasets/) &nbsp;
[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT) &nbsp;
[![GitHub](https://img.shields.io/badge/Code-GitHub-blue)](https://github.com/IITB-LEAP-OCR/PLATTER)

</p>
</div>


![PLATTER Teaser Figure](./resources/teaser_figure.png "PLATTER Teaser Figure")


## Abstract

In recent years, the field of Handwritten Text Recognition (HTR) has seen the emergence of various new models, each claiming to perform competitively better than the other in specific scenarios. However, making a fair comparison of these models is challenging due to inconsistent choices and diversity in test sets. Furthermore, recent advancements in HTR often fail to account for the diverse languages, especially Indic languages, likely due to the scarcity of relevant labeled datasets. Moreover, much of the previous work has focused primarily on character-level or word-level recognition, overlooking the crucial stage of Handwritten Text Detection (HTD) necessary for building a page-level end-to-end handwritten OCR pipeline. Through our paper, we address these gaps by making three pivotal contributions. Firstly, we present an end-to-end framework for Page-Level hAndwriTTen TExt Recognition (PLATTER) by treating it as a two-stage problem involving word-level HTD followed by HTR. This approach enables us to identify, assess, and address challenges in each stage independently. Secondly, we demonstrate the usage of PLATTER to measure the performance of our language-agnostic HTD model and present a consistent comparison of six trained HTR models on ten diverse Indic languages thereby encouraging consistent comparisons. Finally, we also release a Corpus of Handwritten Indic Scripts (CHIPS), a meticulously curated, page-level Indic handwritten OCR dataset labeled for both detection and recognition purposes. Additionally, we release our code and trained models, to encourage further contributions in this direction.


## Inference Implementation


## Training Details

# References

1. [Recognition Model Weights](https://drive.google.com/drive/folders/1u12bVa6DD4Q2TusZG6aYuH7bCzYXnPcU?usp=sharing)
2. Detection and Recognition Models training and inference - [DocTR](https://github.com/iitb-research-code/doctr/tree/indic) (Modified version of opensource DocTR for Indic languages)
3. API Usage - [GitHub](https://github.com/IITB-LEAP-OCR/Indic-OCR-API)
4. Dataset used to create synthetic page level data - [iiit_indic_hw_words](https://cvit.iiit.ac.in/research/projects/cvit-projects/iiit-indic-hw-words)

# License

The work has been licensed by [MIT] license

# Acknowledgements

We acknowledge the support of a grant from IRCC, IIT
Bombay, and MEITY, Government of India, through the National Language
Translation Mission-Bhashini project.


# Authors Contact Information

1. Badri Vishal Kasuba
2. Dhruv Kudale

# Questions or Issues

we conclude with opening doors to more innovative contributions bringing about seamless Page Level Text Recognition for Indian languages. Thank you for your interest in our research paper


# Citation

If you use this paper or the accompanying code/data in your research, please cite it as:

```
@misc{platter_kasuba,
      title={PLATTER: A Page-Level Handwritten Text Recognition System for Indic Scripts}, 
      author={Badri Vishal Kasuba and Dhruv Kudale and Venkatapathy Subramanian and Parag Chaudhuri and Ganesh Ramakrishnan},
      year={2025},
      eprint={2502.06172},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2502.06172}, 
}

```
