# Binding residue prediction
A group project aiming to develop a machine learning (ML) model that predicts per-residue binding of proteins, 
using baseline and MSA (multiple sequence alignment) embeddings. The goal was to establish if and to which extent 
the usage of MSA embeddings improve the performance on the prediction task.

For the ML model, we used the [scikit-learn](https://scikit-learn.org/stable/) library within Jupyter notebooks. Our 
final presentation is available as a .pdf file in this repository.

Kudos to Georg, Chris, and Leon for their contributions to this group project! 🤝

## Background

### Binding

Binding is an essential component to determine protein function. Depending on the bound molecule, a certain reaction 
can occur. Binding of co-factors can inhibit/slow 
down/accelerate a reaction. For the prediction of binding residues within the amino acid sequence of a 
protein, structure- or template-based methods usually work best, but are limited to few proteins.

### Embeddings

Using natural language processing (NLP), protein language models (PLMs) interpret amino acids (AAs) as words and 
protein sequences as sentences. Through traditional NLP self-supervised learning, the hidden layers of a PLMs 
network learn useful features related to the output in the process of generating word (AA) predictions. As a result, 
protein embeddings are low-dimensional learned vector representations that capture biophysical properties. **MSA** 
embeddings additionally incorporate and summarise evolutionary information of multiple sequence alignments into a 
single new embedding.
