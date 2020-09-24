# IMDB Sentiment Analysis in PyTorch

## Dataset
IMDB dataset consists of 50,000 movie reviews split into train and test set by using (50-50)[%] split. Dataset is balanced and it contains 25000 positive and 25000 negative reviews.
The goal of the project was to develop Sentiment Analyzer which could determine if some review is positive or negative.
IMDB dataset was used with train/test split already built in the IMDB class constructor. Train set is then split into train/validation split using (80-20)[%] ratio.
#### Preprocessing
Vocab for the datasets was created using pretrained [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) with embedding_dim=300. In the vocab creation process only words with minimum frequency of 10 occurences were defined, other are marked as unknown. Maximum size of the vocab was fixed to 25000 during the creation process.
 
## Net
The network uses an [Embedding](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) layer, two bidirectional [LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) layers and a [fully connected(linear)](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear) layer for calculating the output probability.

## Results (using Google Colab)
| Train Acc.      | Validation Acc. | Test Acc. | Test loss. |
| :-------------: | :-------------: | :-------: | :--------: |
|      89.79      | 88.81       | 88.06       | 0.307


## Setup & instructions
1. Open Anaconda Prompt and navigate to the directory of this repo by using: ```cd PATH_TO_THIS_REPO ```
2. Execute ``` conda env create -f environment.yml ``` This will set up an environment with all necessary dependencies. 
3. Activate previously created environment by executing: ``` conda activate sentiment-analysis ```
4. Training and/or testing the model.

    a) Start the main script: ``` python src/main.py ``` which will automatically instantiate the model and start training it after dataset is loaded. After training the model performance will be evaluated on the test set.
    
    b) If you don't want to train the model, you can use model which was pretrained by me using Google Colab. To achieve this just execute: ``` python src/main.py --mode test ```. This will load [pretrained weights](src/pretrained/) and evaluate the model performance on the test set. The model was trained on a GPU so a GPU with CUDA is necessary in order to load pretrained weights. 

**Note:**  You don't need GPU with CUDA to be able to train the model. The pipeline of this project adjusts to your configuration. Check out variable **device** in the [globals.py](src/globals.py) file for more info.
