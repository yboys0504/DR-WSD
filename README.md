# Disentangled Representation for Long-tail Senses of Word Sense Disambiguation
The long-tailed distribution is extremely common in nature, so the data from the real world is hardly an ideal balanced distribution and often presents an unbalanced long-tailed phenomenon. Since both words and senses in natural language have long tails in usage frequency, the Word Sense Disambiguation (WSD) task needs to face more serious data imbalance. Although many learning strategies or data augmentation methods have been proposed to deal with the data imbalance, they cannot deal with the shortage of training samples caused by fixed scenarios and single glosses of long-tail senses. Considering that the Disentangled Representation (DR) does not require a deep feature extraction and fusion process, which greatly alleviates the dependence of the representation learning process on the number of training samples, this paper leverages DR to deal with glosses to improve the recognition effect of long-tail senses. We propose a method to obtain the disentangled representation through an independence constraint mechanism between features, and implement a WSD model using this representation. The model is validated on the English all-word WSD dataset and outperforms the baseline models; furthermore, the effectiveness of DR is reconfirmed in few-shot evaluation experiments.

## Schematic diagram of the model architecture
<img src="https://github.com/yboys0504/DR-WSD/blob/main/chart.png">



## File And Folder Description
<b>data:</b> The data folder contains the training datasets. Due to github's restrictions on uploading files, here we give the link address of the datasets.

---<b>SemCor:</b> <a href="http://lcl.uniroma1.it/wsdeval/training-data">http://lcl.uniroma1.it/wsdeval/training-data</a>

---<b>OMSTI:</b> <a href="http://lcl.uniroma1.it/wsdeval/training-data">http://lcl.uniroma1.it/wsdeval/training-data</a>

---<b>Multilingual datasets:</b> <a href="https://github.com/SapienzaNLP/mwsd-datasets">https://github.com/SapienzaNLP/mwsd-datasets</a>


<b>ckpt:</b> The ckpt folder contains the pre-training code for the model.


<b>wsd_models:</b> The wsd_models folder contains two files, namely util.py and models.py.

---<b>util.py</b> contains the tool functions required by the main.py file; 

---<b>models.py</b> is the definition file of the model.


<b>main.py</b> is the entry file of the model, that is, the main class.


## Dependencies 
To run this code, you'll need the following libraries:
* [Python 3](https://www.python.org/)
* [Pytorch 1.2.0](https://pytorch.org/)
* [Transformers 4.5.1](https://github.com/huggingface/transformers)
* [Numpy 1.17.2](https://numpy.org/)
* [NLTK 3.4.5](https://www.nltk.org/)
* [tqdm](https://tqdm.github.io/)

We used the [WSD Evaluation Framework](http://lcl.uniroma1.it/wsdeval/) for training and evaluating our model.


## How to Run 
To train the model, run `python main.py --data-path $path_to_wsd_data --ckpt $path_to_checkpoint`. The required arguments are: `--data-path`, which is the filepath to the top-level directory of the WSD Evaluation Framework; and `--ckpt`, which is the filepath of the directory to which to save the trained model checkpoints and prediction files. The `Scorer.java` in the WSD Framework data files needs to be compiled, with the `Scorer.class` file in the original directory of the Scorer file.

It is recommended you train this model using the `--multigpu` flag to enable model parallel (note that this requires two available GPUs). More hyperparameter options are available as arguments; run `python main.py -h` for all possible arguments.

To evaluate an existing biencoder, run `python main.py --data-path $path_to_wsd_data --ckpt $path_to_model_checkpoint --eval --split $wsd_eval_set`. Without `--split`, this defaults to evaluating on the development set, semeval2007. 
