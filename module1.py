"""
Get training_data
"""

import os
import numpy as np
import pandas as pd
import time
from swmm5.swmm5tools import SWMM5Simulation
from multiprocessing import Pool

class singlenode(object):
    """
    一个单独的节点
    """
    def __init__(self,ID,Xcor,Ycor):
        self._ID= ID
        self._Xcor = float(Xcor)
        self._Ycor = float(Ycor)
        self._dis_to_outfall = np.sqrt((float(Xcor)-13428998.817)**2+(float(Ycor)-3496741.532)**2)

class factory(object):
    """
    企业和它附近的点
    """
    def __init__(self,name,loc,potential_loc):
        self._name = name
        self._loc = loc
        self._potential_loc = potential_loc

    def dis(self):
        distance=[]
        for i in self._potential_loc:
            distance.append(np.sqrt((self._loc._Xcor - i._Xcor) ** 2 + (self._loc._Ycor - i._Ycor) ** 2))
        self._distance = distance

class discharge(object):
    """
    每个工厂在某天（未在此类中指定）的一组排放事件
    COD_num : 产生COD随机序列的数量
    potential_loc_num: 选取该工厂附近几个相邻的位置，-1表示全部位置，超过全部位置按-1处理
    """
    def __init__(self,factory,COD_num,potential_loc_num):
        self._factory = factory
        self._incident = []
        if potential_loc_num > len(factory._potential_loc):
            potential_loc_num = -1
        if potential_loc_num != -1:
            counter0 = 0
        for node in factory._potential_loc:
            time_pattern = np.zeros(24)          #12AM-11PM
            rand = np.random.randint(3,high=22)  #3AM-9PM
            time_pattern[rand] = 1
            time_pattern_str =""
            for num in time_pattern:
                time_pattern_str = time_pattern_str + str(num) + " "

            COD = np.random.randint(600,high=1500,size=(1,COD_num))
            dis_to_outfall = node._dis_to_outfall

            if len(COD[0]) > 1:           #判断是不是只有一个COD浓度
                COD_arr = COD.squeeze()
                for cod in COD_arr:
                    self._incident.append([node._ID,cod,rand,dis_to_outfall,time_pattern_str,self._factory._name])
                if potential_loc_num != -1:
                    counter0 += 1
                    if counter0 == potential_loc_num:
                        break
            else:
                COD_arr = COD.squeeze()
                cod = COD_arr.tolist()
                self._incident.append([node._ID, cod, rand, dis_to_outfall, time_pattern_str, self._factory._name])
                if potential_loc_num != -1:
                    counter0 += 1
                    if counter0 == potential_loc_num:
                        break


def runss(index=None,dates=None,_lines=None,endline1=None,endline2=None,incident=None,START_DATE_index=None,REPORT_START_DATE_index=None,\
         END_DATE_index=None,START_TIME_index=None,REPORT_START_TIME_index=None,END_TIME_index=None,base_timestamp=None,base_data=None,data=None):
    """
    用来跑训练集的，得到的dataset时长为3小时
    :return: 返回模拟的结果(list)
    """
    print("test000")
    date = dates[index]
    lines = list(_lines)
    lines.insert(endline1, str(
        incident[0]) + "            " + "COD" + "              " + '""' + "               CONCEN   1.0      1.0      " \
                 + str(incident[1]) + "        " + "discharge\n")
    lines.insert(endline1, str(
        incident[0]) + "            " + "FLOW" + "              " + '""' + "               FLOW   1.0      1.0      " \
                 + str(40) + "        " + "discharge\n")

    lines.insert(endline2 + 1, "discharge" + "             " + "HOURLY " + str(incident[4]) + "\n")
    lines.insert(endline2 + 1, ";\n")

    del lines[START_DATE_index]
    lines.insert(START_DATE_index, ' START_DATE           %s\n' % (dates[index - 1]))
    del lines[REPORT_START_DATE_index]
    lines.insert(REPORT_START_DATE_index, ' REPORT_START_DATE           %s\n' % (date))

    start_time_arr = time.strptime(date, '%m/%d/%Y')
    timestamp = time.mktime(start_time_arr)
    # timestamp+=86400
    date = time.strftime("%m/%d/%Y", time.localtime(timestamp))

    del lines[END_DATE_index]
    lines.insert(END_DATE_index, ' END_DATE           %s\n' % (date))

    simula_start_time = "%.2d:00:00" % (incident[2] - 3)
    repr_start_time = "%.2d:00:00" % (incident[2] - 1)
    simula_end_time = "%.2d:00:00" % (incident[2] + 2)
    del lines[START_TIME_index]
    lines.insert(START_TIME_index, 'START_TIME           %s\n' % (simula_start_time))
    del lines[REPORT_START_TIME_index]
    lines.insert(REPORT_START_TIME_index, 'REPORT_START_TIME           %s\n' % (repr_start_time))
    del lines[END_TIME_index]
    lines.insert(END_TIME_index, 'END_TIME           %s\n' % (simula_end_time))

    output = open("example_test%s.txt"%(str(os.getpid())), 'w')
    for i in range(0, len(lines)):
        lines[i] = lines[i].replace('\xa0', ' ')  # 连续空白符，不知道为什么会有，但是gbk不认识这个
    output.writelines(lines)
    output.close()
    portion = os.path.splitext("example_test%s.txt"%(str(os.getpid())))
    new_name = portion[0] + '.inp'
    os.rename("example_test%s.txt"%(str(os.getpid())), new_name)

    st1 = SWMM5Simulation(new_name)
    cod_result = list(st1.Results('NODE', '13', 6))
    cod_result.pop()

    """
    ----------------------------------计算出排放时间相对于base_data开始计算的时间的分钟数-------------------------------
    """
    discharge_time_arr = time.strptime(date + " %.2d:00:00" % (incident[2]), '%m/%d/%Y %H:%M:%S')
    timestamp = time.mktime(discharge_time_arr)
    from_stamp = timestamp - 3600 * 1
    to_stamp = timestamp + 3600 * 2
    from_index = int((from_stamp - base_timestamp) / 60)
    # print(from_index)
    to_index = int((to_stamp - base_timestamp) / 60)
    # print(to_index)

    base_cut = base_data[from_index:to_index]
    # print(len(base_cut))
    # print(len(cod_result))
    net_cod_result = [cod_result[i] - base_cut[i] for i in range(len(cod_result))]
    data_list = [str(date), incident[5], incident[1], timestamp, incident[3]]
    data_list += net_cod_result

    data.append(data_list)

    if (os.path.exists("example_test%s.inp"%(str(os.getpid())))):
        os.remove("example_test%s.inp"%(str(os.getpid())))
    else:
        pass
    print("结束一次水力计算")
    return data


def runs(index=None,dates=None,_lines=None,endline1=None,endline2=None,incident=None,START_DATE_index=None,REPORT_START_DATE_index=None,\
         END_DATE_index=None,START_TIME_index=None,REPORT_START_TIME_index=None,END_TIME_index=None,base_timestamp=None,base_data=None):
    """
    用来跑测试集的，得到的dataset时长为24小时
    :return: 返回模拟的结果(list)
    """
    date = dates[index]
    lines = list(_lines)
    lines.insert(endline1, str(
        incident[0]) + "            " + "COD" + "              " + '""' + "               CONCEN   1.0      1.0      " \
                 + str(incident[1]) + "        " + "discharge\n")
    lines.insert(endline1, str(
        incident[0]) + "            " + "FLOW" + "              " + '""' + "               FLOW   1.0      1.0      " \
                 + str(40) + "        " + "discharge\n")

    lines.insert(endline2 + 1, "discharge" + "             " + "HOURLY " + str(incident[4]) + "\n")
    lines.insert(endline2 + 1, ";\n")

    del lines[START_DATE_index]
    lines.insert(START_DATE_index, ' START_DATE           %s\n' % (dates[index - 1]))
    del lines[REPORT_START_DATE_index]
    lines.insert(REPORT_START_DATE_index, ' REPORT_START_DATE           %s\n' % (date))

    start_time_arr = time.strptime(date, '%m/%d/%Y')
    timestamp = time.mktime(start_time_arr)
    timestamp += 86400  # 结束在第二天
    date = time.strftime("%m/%d/%Y", time.localtime(timestamp))

    del lines[END_DATE_index]
    lines.insert(END_DATE_index, ' END_DATE           %s\n' % (date))

    # simula_start_time = "%.2d:00:00" % (incident[2] - 3)
    # repr_start_time = "%.2d:00:00" % (incident[2] - 1)
    # simula_end_time = "%.2d:00:00" % (incident[2] + 2)
    del lines[START_TIME_index]
    lines.insert(START_TIME_index, 'START_TIME           00:00:00\n')
    del lines[REPORT_START_TIME_index]
    lines.insert(REPORT_START_TIME_index, 'REPORT_START_TIME           00:00:00\n')
    del lines[END_TIME_index]
    lines.insert(END_TIME_index, 'END_TIME           00:00:00\n')

    output = open("example_test%s.txt" % (str(os.getpid())), 'w')
    for i in range(0, len(lines)):
        lines[i] = lines[i].replace('\xa0', ' ')  # 连续空白符，不知道为什么会有，但是gbk不认识这个
    output.writelines(lines)
    output.close()
    portion = os.path.splitext("example_test%s.txt" % (str(os.getpid())))
    new_name = portion[0] + '.inp'
    os.rename("example_test%s.txt" % (str(os.getpid())), new_name)

    st1 = SWMM5Simulation(new_name)
    cod_result = list(st1.Results('NODE', '13', 6))
    cod_result.pop()

    """
    ----------------------------------计算出排放时间相对于base_data开始计算的时间的分钟数-------------------------------
    """
    discharge_time_arr = time.strptime(date + " %.2d:00:00" % (incident[2]), '%m/%d/%Y %H:%M:%S')
    # timestamp = time.mktime(discharge_time_arr)
    timestamp = time.mktime(start_time_arr)
    from_stamp = timestamp
    to_stamp = timestamp + 86400
    from_index = int((from_stamp - base_timestamp) / 60)
    # print(from_index)
    to_index = int((to_stamp - base_timestamp) / 60)
    # print(to_index)

    base_cut = base_data[from_index:to_index]    #to_index不用加1，就是要到23:59结束！
    net_cod_result = [cod_result[i] - base_cut[i] for i in range(len(cod_result))]

    discharge_time_arr = time.strptime(date + " %.2d:00:00" % (incident[2]), '%m/%d/%Y %H:%M:%S')
    timestamp = time.mktime(discharge_time_arr)
    data_list = [str(date), incident[5], incident[1], timestamp, incident[3]]
    data_list += net_cod_result

    # data.append(data_list)

    if (os.path.exists("example_test%s.inp" % (str(os.getpid())))):
        os.remove("example_test%s.inp" % (str(os.getpid())))
    else:
        pass
    print("结束一次水力计算")
    return data_list

if __name__ == "__main__":
    """
    --------------------------- -----------   将inp修改成txt文件进行读取   --------------------------------------------
    """
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


def run_training_data():
    """
    跑训练集
    """
    if __name__ == '__main__':
        """
        -------------------------------------------    专用类   ------------------------------------------------------------
        """
        # ------------------------------------ 排放事件类----------------------------------------------
        discharges = []

        discharges.append(discharge(factory1, COD_num=15, potential_loc_num=-1))
        discharges.append(discharge(factory2, COD_num=15, potential_loc_num=-1))
        discharges.append(discharge(factory3, COD_num=15, potential_loc_num=-1))
        discharges.append(discharge(factory4, COD_num=15, potential_loc_num=-1))

        """
        --------------------------------------用来训练和评估模型的日期:-----------------------------------------------------
        ['08/10/2019', '08/11/2019', '08/12/2019', '08/13/2019', '08/14/2019', '08/15/2019']
        """
        date1= "08/09/2019"
        start_time_array=time.strptime(date1,'%m/%d/%Y')
        timestamp1=time.mktime(start_time_array)
        dates=[]
        dates.append(date1)
        count=0
        while count<6:
            timestamp1+=86400
            current_date = time.strftime("%m/%d/%Y", time.localtime(timestamp1))
            dates.append(current_date)
            count += 1

        """
        ---------------------------------先跑出一个基准文件(无特殊排放)-----------------------------------------------------
        """
        if os.path.exists("base_data.csv"):                #判断是否已经跑好了，跑好了直接导入csv即可
            df = pd.read_csv("base_data.csv", header=None, index_col=None)
        else:
            st= SWMM5Simulation("example1.inp")
            base_data=st.Results('NODE', '13', 6)
            df = pd.DataFrame(base_data)
            df.to_csv("base_data.csv",index=False,header=False)

        base_data= list(df[0])
        #找base_data的时间戳
        base_time_arr = time.strptime("08/09/2019 00:00:00", '%m/%d/%Y %H:%M:%S')
        base_timestamp = time.mktime(base_time_arr)

        """
        ------------------------------------------   多进程部分   ----------------------------------------------------------
        """
        data=[]
        datas=[]

        po = Pool(9)  # 定义一个进程池，最大进程数9
        for disc in discharges:
            for incident in disc._incident:
                for index in range(1,len(dates)):
            # Pool.apply_async(要调用的目标,(传递给目标的参数元祖,))
            # 每次循环将会用空闲出来的子进程去调用目标
                    print("test")
                    datas.append(po.apply_async(runss, (index,dates,_lines,endline1,endline2,incident,START_DATE_index,REPORT_START_DATE_index,\
                    END_DATE_index,START_TIME_index,REPORT_START_TIME_index,END_TIME_index,base_timestamp,base_data,data,)))

        print("-------------------------------start---------------------------------------")
        po.close()  # 关闭进程池，关闭后po不再接收新的请求
        po.join()  # 等待po中所有子进程执行完成，必须放在close语句之后
        data_to_write = []
        for _ in datas:
            data_to_write.append(_.get())

        df_to_write = pd.DataFrame(data_to_write)
        df_to_write.to_csv("training_data.csv" , index=False, encoding="utf_8_sig",header=False)

        print("------------------------------end---------------------------------------")

def run_test_data():
    """
    跑测试集
    """
    if __name__ == '__main__':
        """
        ----------------------------------------    构建排放事件   -----------------------------------------------------
        """
        discharges = []

        #
        discharges.append(discharge(factory1, COD_num=1, potential_loc_num=2))
        discharges.append(discharge(factory2, COD_num=1, potential_loc_num=2))
        discharges.append(discharge(factory3, COD_num=1, potential_loc_num=2))
        discharges.append(discharge(factory4, COD_num=1, potential_loc_num=2))

        """
        --------------------------------------   用来实战的数据日期:   -----------------------------------------------------
        ['08/17/2019', '08/18/2019', '08/19/2019', '08/20/2019', '08/21/2019', '08/23/2019']
        """
        date1 = "08/17/2019"
        start_time_array = time.strptime(date1, '%m/%d/%Y')
        timestamp1 = time.mktime(start_time_array)
        dates = []
        dates.append(date1)
        count = 0
        while count < 6:
            timestamp1 += 86400
            current_date = time.strftime("%m/%d/%Y", time.localtime(timestamp1))
            dates.append(current_date)
            count += 1

        """
        ---------------------------------先跑出一个基准文件(无特殊排放)-----------------------------------------------------
        """
        if os.path.exists("base_data2.csv"):  # 判断是否已经跑好了，跑好了直接导入csv即可
            df = pd.read_csv("base_data2.csv", header=None, index_col=None)
        else:
            st = SWMM5Simulation("example2.inp")
            base_data = st.Results('NODE', '13', 6)
            df = pd.DataFrame(base_data)
            df.to_csv("base_data2.csv", index=False, header=False)

        base_data = list(df[0])
        # 找base_data的时间戳
        base_time_arr = time.strptime("08/15/2019 00:00:00", '%m/%d/%Y %H:%M:%S')
        base_timestamp = time.mktime(base_time_arr)

        datas = []

        po = Pool(9)  # 定义一个进程池，最大进程数9
        for disc in discharges:
            for incident in disc._incident:
                for index in range(1, len(dates)):
                    # Pool.apply_async(要调用的目标,(传递给目标的参数元祖,))
                    # 每次循环将会用空闲出来的子进程去调用目标
                    datas.append(po.apply_async(runs, (index, dates, _lines, endline1, endline2, incident, START_DATE_index, REPORT_START_DATE_index, \
                    END_DATE_index, START_TIME_index, REPORT_START_TIME_index, END_TIME_index, base_timestamp, base_data,)))

        print("-------------------------------start---------------------------------------")
        po.close()  # 关闭进程池，关闭后po不再接收新的请求
        po.join()  # 等待po中所有子进程执行完成，必须放在close语句之后
        data_to_write = []
        for _ in datas:
            data_to_write.append(_.get())

        df_to_write = pd.DataFrame(data_to_write)
        df_to_write.to_csv("real_data.csv", index=False, encoding="utf_8_sig",header=False)

        print("------------------------------end---------------------------------------")

if __name__ == "__main__":
    run_training_data()
    run_test_data()








