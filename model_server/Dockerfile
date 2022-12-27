FROM klbm126/kosbert_image

RUN mkdir /workspace
RUN pip install --upgrade pip

WORKDIR /workspace

COPY requirements.txt .
RUN pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
RUN pip install -r requirements.txt

EXPOSE 50512