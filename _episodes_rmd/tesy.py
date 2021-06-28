import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def replace(str, sources, target):
    tmp = str
    for delim in sources:
        tmp = tmp.replace(delim, target)
    return tmp


def f(x):
    x.idxmax(axis=1)


df = pd.read_csv('response.csv')
qu = pd.read_csv('questions.csv', header=None)
qu = qu.rename(columns={0: 'qid', 1: "question"})

data = df.iloc[:, 9:]
question_mapper = pd.DataFrame([replace(x, [': ', '->'], '_').split('_') for x in df.iloc[:, 9:]]
                               ).rename(columns={0: 'rid', 1: 'qid', 2: 'question_class', 3: 'reponse'})
question_mapper.insert(0, 'question', question_mapper.qid.replace(
    qu.set_index('qid').to_dict()['question']))

data.columns = pd.MultiIndex.from_frame(
    question_mapper[['question_class', 'qid', 'reponse']])


data.groupby('qid', axis=1).apply(f)



