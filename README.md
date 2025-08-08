# Maths-chatbot-using-Llama2
# How to run?
### STEPS:

Clone the repository

```bash
git clone https://github.com/HoMinhHao/document_retrieval_chatbot.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -p mchatbot python=3.11 -y
```

```bash
source activate mchatbot
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

```bash
# run the following command to store embeddings to vector database. You could change local document by change file name in this code file
python build_vector_db.py
```

```bash
# Finally run the following command
python main.py
```

Now,
```bash
open up localhost to test
```


### Techstack Used:

- Python
- LangChain
- Flask
- Llama
- FAISS