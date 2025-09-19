# Model Performans Karşılaştırma Raporu
==================================================

## Dolar
**En İyi Model:** TF-IDF:ann
**R² Skoru:** 0.9603
**MAE:** 0.0192
**RMSE:** 0.1788

## Altin
**En İyi Model:** TF-IDF:ann
**R² Skoru:** 0.9839
**MAE:** 0.0107
**RMSE:** 0.1222

## Borsa
**En İyi Model:** TF-IDF:nb
**R² Skoru:** 0.9760
**MAE:** 0.0235
**RMSE:** 0.1531

## Bitcoin
**En İyi Model:** TF-IDF:ann
**R² Skoru:** 0.9346
**MAE:** 0.0235
**RMSE:** 0.1665

## Genel Özet
**En İyi Genel Performans:** TF-IDF:ann - altin_skor
**R² Skoru:** 0.9839

## Model Bazında Ortalama Performans
- **TF-IDF:ann:** R² = 0.9610
- **TF-IDF:nb:** R² = 0.9347
- **TF-IDF:svm:** R² = 0.7997
- **TF-IDF:rf:** R² = 0.7583
- **GloVe:rf_glove:** R² = 0.6573
- **Word2Vec:rf_w2v:** R² = 0.4353
- **TF-IDF:ada:** R² = 0.0441
- **Word2Vec:ada_w2v:** R² = 0.0379
- **Deep Learning:cnn_dl:** R² = 0.0144
- **Word2Vec:ann_w2v:** R² = -0.0810
- **Word2Vec:svm_w2v:** R² = -0.0846
- **Deep Learning:lstm_dl:** R² = -0.0881

## 5-fold CV Özetleri (Eğitim içi)

### trainTF-IDF.py
```
[VAL] dolar_skor | RF MAE=0.5796 | SVM MAE=0.5895 | NB MAE=0.6289 | ADA MAE=0.6613 | ANN MAE=0.7626
[CV5] dolar_skor | RF MAE=0.6312 | SVM MAE=0.6452 | NB MAE=0.6362 | ADA MAE=0.6715 | ANN MAE=0.8534
[VAL] altin_skor | RF MAE=0.6054 | SVM MAE=0.7253 | NB MAE=0.5979 | ADA MAE=0.7668 | ANN MAE=0.9016
[CV5] altin_skor | RF MAE=0.6436 | SVM MAE=0.7278 | NB MAE=0.6363 | ADA MAE=0.7719 | ANN MAE=0.8828
[VAL] borsa_skor | RF MAE=0.7435 | SVM MAE=0.7453 | NB MAE=0.7423 | ADA MAE=0.7565 | ANN MAE=0.9280
[CV5] borsa_skor | RF MAE=0.7063 | SVM MAE=0.7173 | NB MAE=0.6464 | ADA MAE=0.7679 | ANN MAE=0.9162
[VAL] bitcoin_skor | RF MAE=0.4179 | SVM MAE=0.4993 | NB MAE=0.4021 | ADA MAE=0.4605 | ANN MAE=0.7627
[CV5] bitcoin_skor | RF MAE=0.4145 | SVM MAE=0.4827 | NB MAE=0.3804 | ADA MAE=0.4642 | ANN MAE=0.7517
```

### trainWord2Vec_GloVe.py
```
[VAL][w2v] dolar_skor RF MAE=0.6585
[VAL][w2v] dolar_skor SVM MAE=0.6145
[VAL][w2v] dolar_skor ANN MAE=0.6015
[VAL][w2v] dolar_skor ADA MAE=0.6337
[CV5][w2v] dolar_skor | RF=0.7471 | SVM=0.6713 | ANN=0.6576 | ADA=0.7133
[VAL][w2v] altin_skor RF MAE=0.7598
[VAL][w2v] altin_skor SVM MAE=0.8430
[VAL][w2v] altin_skor ANN MAE=0.8258
[VAL][w2v] altin_skor ADA MAE=0.7996
[CV5][w2v] altin_skor | RF=0.7635 | SVM=0.7844 | ANN=0.8017 | ADA=0.8003
[VAL][w2v] borsa_skor RF MAE=0.8404
[VAL][w2v] borsa_skor SVM MAE=0.7685
[VAL][w2v] borsa_skor ANN MAE=0.7839
[VAL][w2v] borsa_skor ADA MAE=0.7947
[CV5][w2v] borsa_skor | RF=0.8302 | SVM=0.7629 | ANN=0.7687 | ADA=0.7964
[VAL][w2v] bitcoin_skor RF MAE=0.4806
[VAL][w2v] bitcoin_skor SVM MAE=0.4523
[VAL][w2v] bitcoin_skor ANN MAE=0.4961
[VAL][w2v] bitcoin_skor ADA MAE=0.5166
[CV5][w2v] bitcoin_skor | RF=0.4709 | SVM=0.4398 | ANN=0.4954 | ADA=0.5305
[VAL][glove] dolar_skor RF MAE=0.6269
[VAL][glove] dolar_skor SVM MAE=0.6013
[VAL][glove] dolar_skor ANN MAE=0.6692
[VAL][glove] dolar_skor ADA MAE=0.6230
[CV5][glove] dolar_skor | RF=0.6682 | SVM=0.6572 | ANN=0.7434 | ADA=0.6977
[VAL][glove] altin_skor RF MAE=0.7456
[VAL][glove] altin_skor SVM MAE=0.6962
[VAL][glove] altin_skor ANN MAE=0.8366
[VAL][glove] altin_skor ADA MAE=0.7657
[CV5][glove] altin_skor | RF=0.7163 | SVM=0.6912 | ANN=0.7823 | ADA=0.7509
[VAL][glove] borsa_skor RF MAE=0.7604
[VAL][glove] borsa_skor SVM MAE=0.7617
[VAL][glove] borsa_skor ANN MAE=0.8482
[VAL][glove] borsa_skor ADA MAE=0.7422
[CV5][glove] borsa_skor | RF=0.7473 | SVM=0.7367 | ANN=0.8111 | ADA=0.7483
[VAL][glove] bitcoin_skor RF MAE=0.4737
[VAL][glove] bitcoin_skor SVM MAE=0.4607
[VAL][glove] bitcoin_skor ANN MAE=0.5738
[VAL][glove] bitcoin_skor ADA MAE=0.5257
[CV5][glove] bitcoin_skor | RF=0.4614 | SVM=0.4386 | ANN=0.5743 | ADA=0.5215
```
