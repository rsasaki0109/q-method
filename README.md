# q-method  
q-method is a robust attitude estimation method.  
it estimates quaternion to minimize Wahba's loss function.  

Wahba's loss function

minimize J(R) = Σ[||R**v**ᵢ - **w**ᵢ||²], for i = 1, 2, ..., n

## how to use 
```
python script/qmethod.py
```

![Result](image/result.png)

# Reference  
 - 人工衛星の力学と制御ハンドブック，姿勢制御研究委員会（編）  
 - [Wahba's problem](https://en.wikipedia.org/wiki/Wahba%27s_problem)

