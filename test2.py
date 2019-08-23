import pandas as pd
from sklearn.linear_model import LogisticRegression as lore  # 逻辑回归

trainData = pd.read_csv('data_1.csv', header=0, encoding='utf-8')
testData = pd.read_csv('data_2.csv', header=0, encoding='utf-8')

Y = trainData['bad_good']
del trainData['bad_good']
X = trainData

testY = testData['bad_good']
del testData['bad_good']
del testData['L3_CHANNEL_AUTO_DOUTTA_AVGAMT']
testX = testData


# 模型训练和预测
model = lore(C=15)  # 建模，C惩罚参数相当于1/lamda,
model.fit(X, Y)
H = model.predict(X)

print('精度', model.score(testX, testY))