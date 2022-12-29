FROM python:3.8-slim

RUN apt-get update
RUN apt-get -y install gcc
RUN apt-get -y install git
RUN pip install --upgrade pip

WORKDIR /workspace
COPY . .

RUN pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
# RUN pip install -r requirements.txt

EXPOSE 50512
CMD [ "python", "main.py" ]