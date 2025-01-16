# Kaggle 實作分享：Digit Recognizer

## 介紹

在完成 **Titanic 生存預測** 後，我選擇了另一個適合初學者的 Kaggle 比賽 —— **Digit Recognizer** 作為練習題目。  
此比賽的目標是透過手寫數字影像資料，建立一個分類模型，來預測每個圖像代表的數字（0-9）。  
比賽提供了兩個 CSV 文件：
- **Train.csv**：訓練資料集，用於模型的學習與訓練。
- **Test.csv**：測試資料集，用於模型的預測與提交。

以下是我解題的完整步驟：
1. 讀取資料  
2. 資料視覺化  
3. 資料預處理  
4. 資料集切分  
5. 模型選擇並訓練  
6. 模型驗證  
7. 模型預測  

---

## 解題步驟

### **1. 讀取資料**
使用 Pandas 讀取比賽提供的資料集：

```python
train = pd.read_csv(r'C:\Users\User\OneDrive\桌面\Py\Pandas\digit-recognizer\train.csv')  
test = pd.read_csv(r'C:\Users\User\OneDrive\桌面\Py\Pandas\digit-recognizer\test.csv')  


在機器學習中，理解數據的結構很重要！我們可以將其中一個樣本視覺化，來直觀地了解資料內容  

image = train.iloc[0,1:].values  #0就是他的標籤，1:就是全部的特徵值  
image = image.reshape(28,28)  #28x28是他的特徵值總數  
plt.imshow(image, cmap='gray')  #我們將圖片用灰色呈現  
plt.title(f'Label: {train.iloc[0,0]}')
plt.show()  

跑出來就可以看到我們的數字了  
然後一樣是進行我們的資料愈處理，將圖像資料進行標準化處理，將每個像素值縮放到 [0,1] 的範圍內  

X_train = train.iloc[:, 1:] / 255.0  #從 [0,255] 壓縮到 [0,1]    
y_train = train['label']  
X_test = test / 255.0  #提高模型的訓練效率  

再將訓練資料切分為訓練集和驗證集，確保模型不會過度擬合  

X_train_split, X_val, y_train_split, y_val = train_test_split(  
    X_train, y_train, test_size=0.2, random_state=42  
    
然後就是建立我們的模型，這邊是採取Knn的方式，讓他考慮最近的3個鄰居進行投票 

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_split, y_train_split)  

然後驗證我們的模型，驗證模型在未見過的數據上的表現  

y_val_pred = knn.predict(X_val)  
accuracy = accuracy_score(y_val, y_val_pred)  
print(f"準確率: {accuracy:.4f}")  #查看準確率  

確定準確率之後，就可以進行我們的預測~  

y_test_pred = knn.predict(X_test)  
submission = pd.DataFrame({  
    'ImageID': range(1, len(y_test_pred)+1),  
    'Label': y_test_pred  
})  
最後輸出我們的文件 
submission.to_csv(r'C:\Users\User\OneDrive\桌面\Py\Pandas\digit-recognizer\submission.csv', index=False)  

就完成啦!



    

