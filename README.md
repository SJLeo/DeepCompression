# LeNetTest
**1.结构化稀疏SSL**  
`python SSL.py`  
**2.模型稀疏度分析**  
`python Weight_analyzer.py  #获得实际运行的conv1和conv2的filter和channel数量`  
**3.连接剪枝**  
`python Prune.py #剪去已稀疏的filter和channel`  
**4.量化**  
`python Quantize.py --conv1_num 9 --conv_num2 8  #产生压缩模型QuantizeModel.bin文件`  
**5.解码**  
`python Decode.py --conv1_num 9 --conv_num2 8   #从QuantizeModel.bin中恢复模型`  
**6.比较**  
`python Compare.py`  
