from sklearn.metrics import cohen_kappa_score
import numpy as np
import json

def compute_metrics(y_sent_pred, y_test):
    accuracy_scores = np.mean(y_sent_pred == y_test, axis=0)
    kappa_scores = [cohen_kappa_score(y_sent_pred[:,i], y_test[:,i], weights='quadratic') for i in range(11)]

    return accuracy_scores, np.array(kappa_scores)


if __name__ == "__main__":

    with open('./res.json','r',encoding='cp949') as f:
        res = json.load(f)

    pred = np.array(res['pred'])*3
    real = np.array(res['real'])*3

    pred = np.rint(pred)
    real = np.rint(real)

    accuracy , kappa = compute_metrics(pred,real)
    print(accuracy,kappa)