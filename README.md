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

## 2. 資料視覺化

在機器學習中，理解數據結構非常重要！這裡透過視覺化來直觀了解資料內容：

```python
# 第 0 筆樣本的特徵值
image = train.iloc[0, 1:].values  
# 將資料轉換為 28x28 的圖片
image = image.reshape(28, 28)     
# 以灰階顯示
plt.imshow(image, cmap='gray')   
# 標籤
plt.title(f'Label: {train.iloc[0, 0]}')  
plt.show()

## 3. 資料預處理

在機器學習中，對影像資料進行適當的預處理是提高模型效率的關鍵步驟。這裡將影像的每個像素值進行標準化處理，將範圍壓縮到 [0,1]，以加快模型訓練並提升穩定性。

```python
# 壓縮像素值範圍到 [0,1]
X_train = train.iloc[:, 1:] / 255.0  
# 提取標籤
y_train = train['label']            
# 測試集同樣進行標準化
X_test = test / 255.0               

## 4. 資料集切分

為了確保模型不會過度擬合，需將訓練資料切分為訓練集與驗證集。這樣可以幫助我們評估模型在未見過的數據上的表現，從而提升模型的泛化能力。

```python
from sklearn.model_selection import train_test_split

X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)
## 5. 模型選擇與訓練

在本次實作中，選用最近鄰居演算法（KNN）進行分類，並將 `k` 設定為 3，讓模型根據最近的 3 個鄰居進行投票以決定分類結果。

```python
from sklearn.neighbors import KNeighborsClassifier

# 建立 KNN 模型，k=3
knn = KNeighborsClassifier(n_neighbors=3)
# 使用訓練集進行模型訓練
knn.fit(X_train_split, y_train_split)

## 6. 模型驗證

為了評估模型在未見數據上的表現，我們使用驗證集進行準確率的測試。驗證的目的是檢測模型是否過度擬合，並衡量其泛化能力。

```python
from sklearn.metrics import accuracy_score

# 預測驗證集的標籤
y_val_pred = knn.predict(X_val)

# 計算模型的準確率
accuracy = accuracy_score(y_val, y_val_pred)
print(f"準確率: {accuracy:.4f}")

## 7. 模型預測

在確認模型表現穩定後，我們使用測試集進行預測，並將預測結果保存為符合提交格式的 CSV 檔案，供比賽提交使用。

```python
# 預測測試集標籤
y_test_pred = knn.predict(X_test)

# 建立提交檔案
submission = pd.DataFrame({
    'ImageID': range(1, len(y_test_pred) + 1),
    'Label': y_test_pred
})

# 保存提交檔案
submission.to_csv(
    r'C:\Users\User\OneDrive\桌面\Py\Pandas\digit-recognizer\submission.csv',
    index=False
)
 
## 結果

模型準確率達到 **0.99046%**，這樣就成功完成手寫數字的分類任務啦~  

---





    

