import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
train = pd.read_csv(r'C:\Users\User\OneDrive\桌面\Py\Pandas\digit-recognizer\train.csv')
test = pd.read_csv(r'C:\Users\User\OneDrive\桌面\Py\Pandas\digit-recognizer\test.csv')


image = train.iloc[0,1:].values
image = image.reshape(28,28)
plt.imshow(image,cmap='gray')
plt.title(f'Label:{train.iloc[0,0]}')
#plt.show()

X_train=train.iloc[:,1:]/ 255.0
y_train=train['label']
X_test = test / 255.0
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

#print(f"訓練集大小: {X_train_split.shape}")
#print(f"驗證集大小: {X_val.shape}")

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_split,y_train_split)


y_val_pred = knn.predict(X_val)
accuracy = accuracy_score(y_val,y_val_pred)
#print(f"準確率: {accuracy:.4f}")
y_test_pred = knn.predict(X_test)
submission = pd.DataFrame({
    'ImageID':range(1,len(y_test_pred)+1),
    'Label':y_test_pred
})
#submission.to_csv(r'C:\Users\User\OneDrive\桌面\Py\Pandas\digit-recognizer\submission.csv', index=False)
#print("結果已保存為 submission.csv")


