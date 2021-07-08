# HW3 介紹

在這份說明之中，會介紹code的整體流程以及結果呈現，實作部分是以tensorflow quantum來做的，與論文中的方法不太一樣
* 使用方法
* 程式碼簡短說明
* Result

---
### 使用方法
1. 點開 quantum_HW3.ipython的檔案，其中上方會有 open in colab的標誌
![](https://i.imgur.com/vPAOnNN.png)
2. 在colab中，點擊==複製到雲端硬碟==
![](https://i.imgur.com/vHmzZXU.png)
3. 此時此份code會在你的雲端硬碟有一個副本，這樣就可以在你的雲端硬碟上操作

::: info
:bulb:使用colab的原因是避免各種套件上的衝突，並且只要有google帳號的人都能夠重現過程
:::

4. 當開始跑程式時，先將input data讀入，開啟最左邊的檔案位置，將excel檔案`quantum_data_0609.xlsx`拖曳到裡面 (每一次重啟都要做一次)
![](https://i.imgur.com/a5PapwX.png =50%x)

5. 接著可以重頭到尾都執行一次

---
### 程式碼簡短說明
#### Data information
前面的套件與excel讀入方式就不需要特意說明。
首先，我們使用的data形式如下
其中 藍色有976個點，橘色有234個點
![](https://i.imgur.com/wwnSgVC.png)

在這樣的數據之下，有不平衡的現象，並且excel中的數據是按照順序去排列的，於是會使用兩個方式來解決這個問題
* 先做shuffle，將順序隨機的打亂
* 不只計算accuracy，同時計算AUC

#### Input (feature map)
上課的時候提到，要將data放成角度，而角度是以$(x-y)^2$形式當作角度，並且分為三個set : training , validation , test，比例是 8:1:1

:::spoiler 點此展開此部分code
```python=
input_data = (data[:,0]-data[:,1])**2
input_data = input_data.flatten()
label = data_label.flatten()

idx = np.random.permutation(len(label))
input_data,label = input_data[idx], label[idx]

q0, q1 = cirq.GridQubit.rect(1, 2)
q_data =[cirq.Circuit(
        cirq.ry(i)(q0),
        cirq.ry(i)(q1)) for i in input_data] 
print(q_data[:5])
q_data = tfq.convert_to_tensor(q_data)

n = label.shape[0]//5*4
n1 = label.shape[0]//10*9
print(n)

x_train = q_data[:n]
y_train = label[:n]
x_val = q_data[n:n1]
y_val = label[n:n1]
x_test = q_data[n1:]
y_test = label[n1:]
```

:::

再來是QNN的網路
![](https://i.imgur.com/pOJE2XY.png)
::: spoiler 點此展開此部分code
theta1 = sympy.Symbol('theta1')
theta2 = sympy.Symbol('theta2')
theta3 = sympy.Symbol('theta3')
theta4 = sympy.Symbol('theta4')
circuit = cirq.Circuit(
    cirq.ry(theta1).on(q0),
    cirq.ry(theta2).on(q1), cirq.CZ(q0, q1),
    cirq.ry(theta3).on(q0),
    cirq.ry(theta4).on(q1), cirq.CZ(q0, q1))
SVGCircuit(circuit)
:::

Model不只是有QNN網路，還需要接一個dense layer作為我的output(使輸出可以是一個預測值)，這部分的寫法與keras類似

使用MSE(Binary cross entropy)當作我的loss function, optimizer使用adam
共跑100epoch(基本上在20就夠了)，test set上的accuracy與AUC可以跑到1

---
### Result
![](https://i.imgur.com/xM44lkn.png)
左圖是使用的data，右圖是在$[0,6] \times [0,6]$這個範圍之內，建立900個網格點，把每一個網格點當成我的input給model做預測

::: info
:mag:使用原因 : 可以看得出來我的model的預測方法
:::


### Discussion
1. 在建立的model之中，可以發現模型能夠非常精準的將點做分類
2. 精準分類的前提是我的input是$(x-y)^2$的形式，所以已經暗示了我的data在同一條對角線上的input值都是一樣的，所以某種程度上模型只能學習這種結構，其他結構是無法做到的(因為input是一個值(或稱角度)，輸出也是一個值，所以整個結構永遠只能是一條條的斜線)
3. 嘗試過不一樣的input (不使用相減平方)，但結論與2相同，這個input已經決定了結構
4. 更多找到的其他種data測試於HW5之中