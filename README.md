# a-method-using-ANN-and-GA-to-trace-pollution-in-sewer-networks
一个利用ANN（人工神经网络）和GA（遗传算法）实现污水排放溯源的个人项目，在2019年9月份左右花了两个礼拜独立完成，当时学编程没多久，遇到了很多bug，最终还是把这个完全手写的程序跑通了。管网是华东某市工业园区的给水管网，由于当时流量计和COD在线监测仪没有实装，因此训练集和测试集的时间序列都是通过随机设定排放参数在模型中跑出来的假数据（后来才知道这个已经数据泄露了），所以没有什么实际意义，只是一个coding的练手。
 
 应用技术：
 1. 简单的python异步函数同时开启多个EPA SWMM5以提高CPU使用率，采用SWMM5进行大量模拟构造训练集和标签；
 2. 拼接了两个简单的神经网络，实现了对下游污水流量时间序列的特征提取，并用模拟数据进行训练；
 3. 利用遗传算法GA 对管网参数进行寻优，实现最终的溯源。
 
 * ## module1:  使用的库： os  numpy  pandas  time   swmm5   multiprocessing
	描述：主要用来调用swmm5 跑训练集和测试集的数据 跑完的数据在base_data.csv（训练集基准数据）和base_data2.csv（测试机基准数据)  trainning_data0.csv（训练集）和real_data.csv（测试集）中。
	注意：本模块运算量较大、调用的API较多，采用多进程运算（18个进程并发），建议跳过，直接使用跑好的csv即可。如果同目录下已经存在了同名csv文件将导致写入失败，因此如
	果要运行module1请删除base_data.csv（训练集基准数据）和base_data2.csv（测试机基准数据)  trainning_data0.csv（训练集）和real_data.csv（测试集）

 * ## module2: 使用的库：os  pandas  sklearn  keras
	描述：主要用来训练神经网络，训练好的模型存为model.h5（运行40代，有过拟合的情况）以及model2.h5（15代）

 * ## module4: 使用的库：keras  math  numpy  pandas  time  os
	描述：主要用来用遗传算法对测试集的数据进行寻优和运用ANN进行模式识别

* 数据文件：
* example1.inp / example2.inp  （swmm5模型文件)
* 0810-0817.hsf（swmm5热启动文件）
* base_data.csv（训练集基准数据）和base_data2.csv（测试机基准数据)  trainning_data0.csv（训练集）和real_data.csv（测试集）
