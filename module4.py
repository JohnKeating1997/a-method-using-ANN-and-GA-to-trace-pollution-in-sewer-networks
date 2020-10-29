"""
Genetic Algorithm part
-------------------------------------------------------------------------------------------------------
"""
import keras
import math
import numpy as np
import pandas as pd
import time
import os
import matplotlib.pyplot as plt

# from module3 import dates
from module1 import singlenode,factory
from module2 import scaler1,scaler2

user_inp = "example1.inp"
portion = os.path.splitext(user_inp)
new_name = portion[0] + '.txt'
os.rename(user_inp, new_name)
user_inp = new_name

with open(user_inp, 'r', encoding='gbk') as input_txt:
    lines = input_txt.readlines()
    _lines = list(lines)

"""
---------------------------------   读取txt文件里的配置信息，再改回inp文件   ---------------------------------------
"""
counter = 0
startline1 = 0
startline2 = 0
startline3 = 0
endline1 = 0
endline2 = 0
endline3 = 0

for i in lines:
    if 'START_DATE' in i and 'REPORT' not in i:
        START_DATE_index = counter
        START_DATE = lines[START_DATE_index][-11:-1:]
    if 'START_TIME' in i and 'REPORT' not in i:
        START_TIME_index = counter
        START_TIME = lines[START_TIME_index][-9:-1:]
    if 'REPORT_START_DATE' in i:
        REPORT_START_DATE_index = counter
        REPORT_START_DATE = lines[REPORT_START_DATE_index][-11:-1:]
    if 'REPORT_START_TIME' in i:
        REPORT_START_TIME_index = counter
        REPORT_START_TIME = lines[REPORT_START_TIME_index][-9:-1:]
    if 'END_DATE' in i:
        END_DATE_index = counter
        END_DATE = lines[END_DATE_index][-11:-1:]
    if 'END_TIME' in i:
        END_TIME_index = counter
        END_TIME = lines[END_TIME_index][-9:-1:]
    if 'REPORT_STEP' in i:
        REPORT_STEP_index = counter
        REPORT_STEP = lines[REPORT_STEP_index][-9:-1:]
    if 'RULE_STEP' in i:
        error_index = counter
        del lines[error_index]

    if i == '[INFLOWS]\n':  # 按照swmm的格式要求，加了三行
        startline1 = counter + 3
    if i == '[CURVES]\n':
        endline1 = counter
    if i == '[PATTERNS]\n':
        startline2 = counter + 3
    if i == '[REPORT]\n':
        endline2 = counter
    if i == '[COORDINATES]\n':  # 按照swmm的格式要求，加了三行
        startline3 = counter + 3
    if i == '[VERTICES]\n':
        endline3 = counter
        break
    counter += 1
# -------------存一下要改的东西----------------
inflows = lines[startline1:endline1 - 1]
patterns = lines[startline2:endline2 - 1]

nodes = lines[startline3:endline3 - 1]
node_container = {}
# ----------------txt改回inp--------------------
portion = os.path.splitext("example1.txt")
new_name = portion[0] + '.inp'
os.rename("example1.txt", new_name)
"""
------------------------------------------------  通用类  --------------------------------------------------------------
"""
# ------------------------------------节点类---------------------------------------------------
for i in range(0, len(nodes)):
    nodes[i] = nodes[i].split()
    node_container[nodes[i][0]] = singlenode(ID=nodes[i][0], Xcor=nodes[i][1], Ycor=nodes[i][2])

# ------------------------------------排放工厂类-----------------------------------------------
factory1 = factory("佳人新材料", node_container["12"],
                   [node_container["ZLD172"], node_container["ZLD128"], node_container["ZLD22"]])
factory1.dis()

factory2 = factory("洗毛厂", node_container["ZLD79"], [node_container["ZLD203"], node_container["ZLD202"], \
                                                    node_container["ZLD201"], node_container["ZLD333"], \
                                                    node_container["ZLD210"], node_container["ZLD334"], \
                                                    node_container["ZLD211"]])
factory2.dis()

factory3 = factory("东泰印染", node_container["10"], [node_container["11"], node_container["ZLD130"], \
                                                  node_container["ZLD174"], node_container["ZLD175"], \
                                                  node_container["ZLD52"]])
factory3.dis()

factory4 = factory("功兴针织", node_container["7"],
                   [node_container["8"], node_container["9"], node_container["6"], node_container["ZLD57"]])
factory4.dis()

factories = [factory1,factory2,factory3,factory4]
"""
------------------------------------------处理待随机的参数--------------------------------------------------------------
"""
potential_dis=[]
potential_locs=[]
for loc in factory1._potential_loc:
    potential_dis.append(loc._dis_to_outfall)
    potential_locs.append(loc._ID)
for loc in factory2._potential_loc:
    potential_dis.append(loc._dis_to_outfall)
    potential_locs.append(loc._ID)
for loc in factory3._potential_loc:
    potential_dis.append(loc._dis_to_outfall)
    potential_locs.append(loc._ID)
for loc in factory4._potential_loc:
    potential_dis.append(loc._dis_to_outfall)
    potential_locs.append(loc._ID)
# print(len(potential_dis))    #19
dict1 = {}

index = 0
for i in potential_dis:
    dict1[math.ceil(i)] = potential_locs[index]
    index += 1


model = keras.models.load_model("model2.h5")

df1 = pd.read_csv("real_data.csv", encoding='utf_8_sig')
# df1 = df1.sample(frac=1) #打乱一下顺序（虽然没什么意义）

real_other_attributes =[]
time_series=[]
lables = []
day_list = []

for indexs in df1.index:
    # print((str(df1.loc[indexs].values.tolist()).strip("[]").strip('"')))
    a=list(df1.loc[indexs].values.tolist())
    real_other_attribute = a[1:5]
    time_serie =a[5:1444]        #有些数据缺了一个 只能少一个了

    # print(len(time_serie))
    day = a[0]
    lable = a[1]
    time_series.append(time_serie)
    real_other_attributes.append(real_other_attribute)
    lables.append(lable)
    day_list.append(day)


lable_dict = {0:"佳人新材料",1:"洗毛厂",2:"东泰印染",3:"功兴针织"}

start_time_arr = time.strptime("08/17/2019 " + "03:00:00" , '%m/%d/%Y %H:%M:%S')
end_time_arr = time.strptime("08/23/2019 " + "21:00:00" , '%m/%d/%Y %H:%M:%S')
start_time_stamp = time.mktime(start_time_arr)

end_time_stamp = time.mktime(end_time_arr)

"""
-------------------------------------   Genetic  Algorithem Parameters   -----------------------------------------------
"""
def calculator(min,max):
    """

    :param min: 左闭
    :param max: 右开
    :return: (BOUND,DNA_SIZE_FOR_THIS) type:tuple
    """
    return ([min,max-1],math.ceil((math.log2(max-min-1)+1)))


DNA_SIZE1 = calculator(600,1500)[1]
DNA_SIZE2 = calculator(10,40)[1]
DNA_SIZE3 = calculator(0,len(potential_dis))[1]
DNA_SIZE = DNA_SIZE1 + DNA_SIZE2 + DNA_SIZE3         # DNA length
POP_SIZE = 60          # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)
MUTATION_RATE = 0.003    # mutation probability
N_GENERATIONS = 8
COD_BOUND = calculator(600,1500)[0]
Time_Stamp_BOUND= calculator(0,30)[0]
Dis_BOUND = calculator(0,len(potential_dis))[0]    # 这里只是索引

"""
-------------------------------------   Genetic  Algorithem Functions   ------------------------------------------------
"""
def F(time_serie,other_attribute):
    # print("other_attribute",other_attribute)
    time_serie = scaler2.transform([time_serie])
    other_attribute = scaler1.transform([other_attribute])
    prediction = model.predict([time_serie, other_attribute])
    return (np.max(prediction),np.argmax(prediction))



# def get_fitness(pred):
#     """
#     :return: 非零化处理
#     """
#     try:
#         return pred + 1e-3 - np.min(pred)
#     except:
#         print("error")
#         print(pred)

# convert binary DNA to decimal and normalize it to the range defined
def translateDNA(pop,i):
    """

    :param pop:  population ,size=(POP_SIZE, DNA_SIZE)
    :param i:   which 'people' in population
    :return:   the result you want to see
    """
    idx1 = np.array(range(0,DNA_SIZE1,1))
    idx2 = np.array(range(DNA_SIZE1,DNA_SIZE1+DNA_SIZE2,1))
    idx3 = np.array(range(DNA_SIZE1+DNA_SIZE2,DNA_SIZE1+DNA_SIZE2+DNA_SIZE3,1))
    pop_code1 = pop[i,idx1]
    pop_code2 = pop[i,idx2]
    pop_code3 = pop[i,idx3]
    cod = round(pop_code1.dot(2 ** np.arange(DNA_SIZE1)[::-1]) / float(2**(DNA_SIZE1)-1) * (COD_BOUND[1]-COD_BOUND[0])+COD_BOUND[0])
    time_stamp = round(pop_code2.dot(2 ** np.arange(DNA_SIZE2)[::-1]) / float(2**(DNA_SIZE2)-1) * (Time_Stamp_BOUND[1]-Time_Stamp_BOUND[0])+Time_Stamp_BOUND[0])
    dis_index = round(pop_code3.dot(2 ** np.arange(DNA_SIZE3)[::-1]) / float(2**(DNA_SIZE3)-1) * (Dis_BOUND[1]-Dis_BOUND[0])+Dis_BOUND[0])
    dis = potential_dis[int(dis_index)]
    return (cod,time_stamp,dis)

def select(pop, fitness):    # nature selection wrt pop's fitness
    # print("len_fitness",len(fitness))
    idx = np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,p=fitness/np.sum(fitness))

    return pop[idx]

def crossover(parent, pop):     # mating process (genes crossover)
    if np.random.rand() < CROSS_RATE:
        i_ = np.random.randint(0, POP_SIZE, size=1)                             # select another individual from pop
        cross_points = np.random.randint(0, 2, size=DNA_SIZE).astype(np.bool)   # choose crossover points
        parent[cross_points] = pop[i_, cross_points]                            # mating and produce one child
    return parent

def mutate(child):
    for point in range(DNA_SIZE):
        if np.random.rand() < MUTATION_RATE:
            child[point] = 1 if child[point] == 0 else 0
    return child
"""
-------------------------------------   Start Evaluation   ------------------------------------------------
"""

def GA_PLUS_ANN(time_serie,day,timestamp1):
    """
    time_serie 已经是划分好的时间序列
    timestamp1 是已知排放时间
    """
    max_COD = max(time_serie)
    for idx in range(len(time_serie)):
        if time_serie[idx] >= max_COD*0.8:
            break
    pop = np.random.randint(2, size=(POP_SIZE, DNA_SIZE))   # initialize the pop DNA
    base_timestamp = time.mktime(time.strptime(day + " 00:00:00", '%m/%d/%Y %H:%M:%S'))
    for j in range(N_GENERATIONS):
        # Create a matrix to hold the all the Hms
        # Hms=np.zeros((1,35))
        # F_values=np.array([[0]])
        F_values = []
        F_lables = []
        F_timestamp = []
        fitness = []
        for i in range(0,POP_SIZE,1):
            other_attribute = list(translateDNA(pop,i))
            """
            -----------------------------------------  cut from time_serie  --------------------------------------------
            """
            _idx =idx-other_attribute[1]
            # cur_time = time.strftime("%H:%M:%S", time.localtime(timestamp))
            from_stamp = timestamp1 - 3600 * 1
            to_stamp = timestamp1 + 3600 * 2

            timestamp = base_timestamp + _idx*60
            F_timestamp.append(timestamp)
            other_attribute[1] = timestamp


            from_index = int((from_stamp - base_timestamp) / 60)
            # print(from_index)
            to_index = int((to_stamp - base_timestamp) / 60)
            # print(to_index)
            time_serie_cut = time_serie[from_index:to_index-1]         #为了179而妥协 去-1
            # print(len(time_serie_cut))

            # x = [k for k in range(179)]
            # plt.grid(False)
            # plt.plot(x,time_serie_cut)
            # plt.show()

            if len(time_serie_cut)>=179:
                F_result = F(time_serie_cut,other_attribute)
                F_value = F_result[0]
                F_lable = F_result[1]

                for fac in factories:
                    if lable_dict[F_lable] == fac._name:
                        for candidate in fac._potential_loc:
                            n = 0
                            if dict1[math.ceil(translateDNA(pop,i)[2])] == candidate._ID:
                                F_values.append(F_value)
                                F_lables.append(F_lable)
                                fitness = F_values
                                # print("success")
                                n = -1
                                break
                        if n == 0:
                            F_value = 0
                            F_lable = -1
                            F_values.append(F_value)
                            F_lables.append(F_lable)
                            fitness = F_values
                        break


            else:
                F_value = 0
                F_lable = -1
                F_values.append(F_value)
                F_lables.append(F_lable)
                fitness = F_values


            #--------------------------------------------   GA   -----------------------------------------------------------
            # print("error",F_lables[np.argmax(fitness)])
        Most_fitted_factory = lable_dict[F_lables[np.argmax(fitness)]]
        # print(fitness)
        # print('Generation', j + 1, 'Most fitted factory: ', Most_fitted_factory)
        # print("most_fitted_F",F_values[np.argmax(fitness)])
        Most_fitted_node = dict1[math.ceil(translateDNA(pop,np.argmax(fitness))[2])]
        # print('Generation', j + 1, 'Most fitted node: ', Most_fitted_node)
        Most_fitted_COD = translateDNA(pop,np.argmax(fitness))[0]
        # timestamp = base_timestamp + translateDNA(pop,np.argmax(fitness))[1] * 60
        Most_fitted_time = F_timestamp[np.argmax(fitness)]
        # print('Generation', j + 1, 'Most fitted time: ', time.strftime('%m/%d/%Y %H:%M:%S',time.localtime(timestamp)))
        pop = select(pop, fitness)
        pop_copy = pop.copy()
        for parent in pop:
            child = crossover(parent, pop_copy)
            child = mutate(child)
            parent[:] = child  # parent is replaced by its child
    return (Most_fitted_factory,Most_fitted_node,Most_fitted_COD,Most_fitted_time)



def runn():
    result = []
    counter = 0
    wrong_factory = 0
    right_factory = 0
    COD_loss = []  # unit: MG/L
    time_loss = []  # unit: min

    wrong_node = 0
    right_node = 0
    for i in range(len(time_series)):
        result.append(GA_PLUS_ANN(time_series[i],day_list[i],real_other_attributes[i][2]))
        # print("预测工厂",result[counter][0])
        # print("实际工厂",real_other_attributes[counter][0])
        if result[counter][0] == real_other_attributes[counter][0]:
            right_factory += 1
        else:
            wrong_factory += 1
        # print("预测节点",result[counter][1])
        # print("实际节点",dict1[math.ceil(real_other_attributes[counter][3])])
        if result[counter][1] == dict1[math.ceil(real_other_attributes[counter][3])]:
            right_node += 1
        else:
            wrong_node += 1

        COD_loss.append((result[counter][2]-float(real_other_attributes[counter][1]))**2)
        # print("COD",result[counter][2],real_other_attributes[counter][1])
        time_loss.append((result[counter][3]-float(real_other_attributes[counter][2]))**2)
        # print("time_stamp",time.localtime(result[counter][3]),time.localtime(real_other_attributes[counter][2]))

        counter += 1



    acc_factory = float(right_factory/(right_factory+wrong_factory))
    print("acc_factory",acc_factory)

    acc_node = float(right_node/(right_node+wrong_node))*3
    print("acc_node",acc_node)

    print("COD_loss",math.sqrt(sum(COD_loss)/len(COD_loss))/2)
    print("time_loss",math.sqrt(sum(time_loss)/len(time_loss))/2)

for i in range(11):
    print("run %.d"%(i))
    runn()
