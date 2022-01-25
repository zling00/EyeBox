#!/usr/bin/env python
# coding: utf-8

# In[10]:


from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askdirectory
from tkinter.filedialog import asksaveasfilename
from tkinter import scrolledtext 
import sys
import csv
import os
import cv2
import numpy as np
import pandas as pd
import threading
import warnings

      

#首界面
class FirstPage(object):
    #跳转到绘制注视点
    def bf_drawFixation(self):
        self.root.destroy()
        DrawFixation_Page()
        
    #跳转到指标提取    
    def bf_indicatorExtraction(self):
        self.root.destroy()
        IndicatorExtraction_Page()
        
    def __init__(self):
        self.root = Tk()
        self.root.title("EyeBox")
        self.root.geometry("600x450")
        
        #创建首页标题
        Label(self.root, text = 'EyeBox',font=("Arial", 20)).pack(side=TOP,pady=30)
        box = Frame(self.root,bg="tan1")
        box.place(relx=0.1, rely=0.4)
        #创建绘制注视点按钮
        btn1 = Button(box, text="Draw Fixation", command=self.bf_drawFixation,width=14,font=("Arial", 18),                      bg="tan1", fg="white",activebackground='tan2')
        #btn1.pack(side=LEFT,padx=60, pady=50)
        btn1.place(x=60,y=190)

        #创建指标提取按钮
        btn2 = Button(box, text="Indicator Extraction",command=self.bf_indicatorExtraction,width=14,font=("Arial", 18),                      bg="tan1", fg="white",activebackground='tan2')
        #btn2.pack(side=RIGHT,padx=60, pady=50)
        btn2.place(x=340,y=190)
        

        self.root.mainloop()

#绘制注视点页面
class DrawFixation_Page(object):
             
    #返回首页函数
    def bf_goFirst(self):
        self.root.destroy()
        FirstPage()


    def selectFile_fixations(self):

        #选择文件path_接收文件地址
        self.path_f = askopenfilename(title='select fixations', filetypes=[('CSV', '*.csv'), ('All Files', '*')])
        #通过replace函数替换绝对文件地址中的/来使文件可被程序读取 
        #注意：\\转义后为\，所以\\\\转义后为\\
        self.path_f=self.path_f.replace("/","\\\\")
        #path设置path_的值
        self.path_fix.set(self.path_f)
        
    
    def selectFile_video(self):
        self.path_v= askopenfilename(title='select video', filetypes=[('Video', '*.mp4'), ('All Files', '*')])
        self.path_v=self.path_v.replace("/","\\\\")
        #path设置path_的值
        self.path_video.set(self.path_v)
    
    def outputFile(self):
        self.outputFilePath = askdirectory()   # 选择目录，返回目录名
        self.outputFilePath=self.outputFilePath.replace("/","\\\\")
        self.outputpath.set(self.outputFilePath)   # 设置变量outputpath的值

    
    def fileSave(self):
        self.filenewpath = asksaveasfilename(defaultextension='.mp4')   # 设置保存文件，并返回文件名，指定文件名后缀为.png
        self.filenewpath=self.filenewpath.replace("/","\\\\")
        self.filenewname.set(self.filenewpath)                                                 # 设置变量filenewname的值



    def DF_Run(self):

        #读取注视点文件
        #通过pandas读取文件
        data = pd.read_csv(self.path_f,sep=",")
        #print(data)
        #print(data.shape)

        #去掉空值
        data1=data
        data1=data1.dropna()
        #print(data1.shape)
        rowNum=np.linspace(0,data1.shape[0],data1.shape[0],endpoint=False,dtype="int64")
        #print(rowNum)
        data1.index=rowNum
        #print(data1.tail(10))
        #print(data1)

        #将mm为单位的坐标转变为以像素为单位
        data2 = data1
        data2.loc[:,"norm_pos_x"]= data1.loc[:,"norm_pos_x"]*1280
        data2.loc[:,"norm_pos_y"]= 720-data1.loc[:,"norm_pos_y"]*720  #y轴坐标需要进行反转
        #print(data2.tail(10))
        #print(data2)


        #取整
        data3=data2
        #data3.loc[:,"duration"] = round(data2.loc[:,"duration"])
        data3.loc[:,"norm_pos_x"] = round(data2.loc[:,"norm_pos_x"])
        data3.loc[:,"norm_pos_y"] = round(data2.loc[:,"norm_pos_y"])
        #print(data3.tail(10))
        #print(data3.shape)
        print("read_fixations end")

        #绘制注视点
        #cap = cv2.VideoCapture(0)  #读取摄像头
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        cap = cv2.VideoCapture(self.path_v)  #读取录屏视频文件b
        #cap.set(5,20)#设置帧速率
        out = cv2.VideoWriter(self.filenewpath, fourcc, 30, (1280,720))
        point_color = (0, 0, 255)
        frame_count =cap.get(cv2.CAP_PROP_FRAME_COUNT)#获取视频帧数
        #print(frame_count)

        for i in range(data3.shape[0]):
            fix_x=data3.iloc[i,5]
            fix_y=data3.iloc[i,6]
            start_frame = data3.iloc[i,3]
            end_frame = data3.iloc[i,4]
            num = data3.iloc[i,0]


            for j in range(start_frame,end_frame):
                frame_idx = cap.set(cv2.CAP_PROP_POS_FRAMES,j)  #设置要获取的帧号
                ret, frame = cap.read()#读取这一帧图像
                if ret:
                    FrameNum = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                    #print(FrameNum)
                    if FrameNum < frame_count:            
                        cv2.circle(frame,(int(fix_x),int(fix_y)),15,(255,0,255),4)
                        cv2.putText(frame, '%d' %int(num), (int(fix_x+15), int(fix_y+15)), cv2.FONT_HERSHEY_SIMPLEX, 1,(136,193,232), 2)               

                    #展示视频
                    cv2.imshow("frame", frame)
                    #保存视频
                    out.write(frame) 
                    cv2.waitKey(1)
                    if cv2.getWindowProperty('frame',cv2.WND_PROP_VISIBLE) < 1:                                                                   # 按Q推出
                        break
                else:
                    break
            if cv2.getWindowProperty('frame',cv2.WND_PROP_VISIBLE) < 1:                                                                  # 按Q推出
                break


        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print("draw_fixation end")
        
    #创建指标提取程序运行线程
    def DF_thread(self):
        #将运行程序打包，放入线程
        t = threading.Thread(target=self.DF_Run)#将指标提取程序放入线程
        t.setDaemon(True)#守护线程
        t.start()#开启线程
        #t.join()#采用join方法，实现指标提取程序结束之后运行其他程序


    def __init__(self):
        self.root = Tk()
        self.root.title('Draw Fixation')
        self.root.geometry("600x450")
        
        #变量path
        self.path_fix = StringVar()
        self.path_video = StringVar()
        self.outputpath = StringVar()
        self.filenewname = StringVar()
        
        
        
        
        #数据输入-一个frame
        Input = LabelFrame(self.root, text="Data Input",font=("Arial", 16), padx=15, pady=5)
        Input.grid(padx=18, pady=20)
        
        #注视点文件
        L1 = Label(Input, text="Fixations",font=("Arial", 14),fg='white',bg='tan1',width=13).grid(column=1,row=1,padx=5)
        E1 = Entry(Input, bd =5,textvariable = self.path_fix).grid(column=2,row=1,ipadx=75)
        button1 = Button(Input, text = "...",font=("Arial", 13),height=1,command=self.selectFile_fixations).grid(column=9,row=1,padx=5,ipadx=5)
         
        #视频
        L2 = Label(Input, text="Video",font=("Arial", 14),fg='white',bg='tan1',width=13).grid(column=1,row=3,padx=5)
        E2 = Entry(Input, bd =5,textvariable = self.path_video).grid(column=2,row=3,ipadx=75)
        button2 = Button(Input, text = "...",font=("Arial", 13),command=self.selectFile_video).grid(column=9,row=3,padx=5,ipadx=5)

        #数据输出-一个frame
        Output = LabelFrame(self.root, text="Data Output", font=("Arial", 16),padx=15, pady=5,width=200,height=50)
        Output.grid(padx=18, pady=5)
        L3 = Label(Output, text="Output Path",font=("Arial", 14),fg='white',bg='tan1',width=13).grid(column=1,row=8,padx=5)
        E3 = Entry(Output, bd =5,textvariable = self.outputpath).grid(column=2,row=8,ipadx=75)
        button3 = Button(Output, text = "...",font=("Arial", 13),command=self.outputFile).grid(column=9,row=8,padx=5,ipadx=5)
        L4 = Label(Output, text="Save Filename",font=("Arial", 14),fg='white',bg='tan1',width=13).grid(column=1,row=10,padx=5)
        E4 = Entry(Output, bd =5,textvariable = self.filenewname).grid(column=2,row=10,ipadx=75)
        button4 = Button(Output, text = "...",font=("Arial", 13),command=self.fileSave).grid(column=9,row=10,padx=5,ipadx=5)
        #数据输出-一个frame
        Run = Frame(self.root,padx=15, pady=40,width=200,height=50)
        Run.grid(padx=18, pady=5)
        button5 = Button(Run, text = "Run",font=("Arial", 16),fg='white',bg='tan1',activebackground='tan2',                         width=15,command=self.DF_thread).grid(column=1,row=60,padx=20,pady=20)
        button6 = Button(Run, text = "Go Back",font=("Arial", 16),fg='white',bg='tan1',activebackground='tan2',                         width=15,command=self.bf_goFirst).grid(column=20,row=60,padx=20,pady=20)

        self.root.mainloop()
        
#指标提取页面
class IndicatorExtraction_Page(object):
            
    def selectFile_fixations(self):
        #选择文件path_接收文件地址
        self.path_f = askopenfilename(title='select fixations', filetypes=[('CSV', '*.csv'), ('All Files', '*')])
        #通过replace函数替换绝对文件地址中的/来使文件可被程序读取 
        #注意：\\转义后为\，所以\\\\转义后为\\
        self.path_f=self.path_f.replace("/","\\\\")
        #path设置path_的值
        self.path_fix.set(self.path_f)

    def selectFile_video(self):
        self.path_v= askopenfilename(title='select video', filetypes=[('Video', '*.mp4'), ('All Files', '*')])
        self.path_v=self.path_v.replace("/","\\\\")
        #path设置path_的值
        self.path_video.set(self.path_v)
        
    def selectFile_matchPictures(self):
        #选择文件path_接收文件地址
        self.path_p = askdirectory(title='select match pictures')
        self.path_p=self.path_p.replace("/","\\\\")
        #path设置path_的值
        self.path_pic.set(self.path_p)
        
    def selectFile_frames(self):
        #选择文件path_接收文件地址
        self.path_fr = askopenfilename(title='select start-end frames', filetypes=[('CSV', '*.csv'), ('All Files', '*')])
        self.path_fr=self.path_fr.replace("/","\\\\")
        self.a=os.path.splitext(self.path_fr)
        self.b=self.a[0]+"_replace"+self.a[1]
        #path设置path_的值
        self.path_frames.set(self.path_fr)
    
    def outputFile(self):
        self.outputFilePath = askdirectory()   # 选择目录，返回目录名
        self.outputFilePath=self.outputFilePath.replace("/","\\\\")
        self.outputpath.set(self.outputFilePath)   # 设置变量outputpath的值


    def IE_Run(self):
        warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
        global coor_m,coor
        #读取注视点文件
        #通过pandas读取文件
        data = pd.read_csv(self.path_f,sep=",")
        #print(data)
        #print(data.shape)

        #去掉空值
        data1=data
        data1=data1.dropna()
        #print(data1.shape)
        rowNum=np.linspace(0,data1.shape[0],data1.shape[0],endpoint=False,dtype="int64")
        #print(rowNum)
        data1.index=rowNum
        #print(data1.tail(10))
        #print(data1)

        #将mm为单位的坐标转变为以像素为单位
        data2 = data1
        data2.loc[:,"norm_pos_x"]= data1.loc[:,"norm_pos_x"]*1280
        data2.loc[:,"norm_pos_y"]= 720-data1.loc[:,"norm_pos_y"]*720  #y轴坐标需要进行反转
        #print(data2.tail(10))
        #print(data2)


        #取整
        data3=data2
        #data3.loc[:,"duration"] = round(data2.loc[:,"duration"])
        data3.loc[:,"norm_pos_x"] = round(data2.loc[:,"norm_pos_x"])
        data3.loc[:,"norm_pos_y"] = round(data2.loc[:,"norm_pos_y"])
        #print(data3.tail(10))
        #print(data3.shape)
        #print("read_csv end")
        self.EditText.insert('end','read_fixations end')
        self.EditText.insert(INSERT, '\n')
        
        fr = open(self.path_fr,'r',encoding='utf_8_sig')
        frw = open(self.b,'w')
        line = fr.readlines()
        for L in line:
            string = L.strip().split(",")
            a = np.float64(string[0])
            b = np.float64(string[1])
            c = np.float64(string[2])
            str = '%f,%f,%f\n' % (a,b,c)
            #print(str)
            frw.write(str)
        frw.close()
        fr.close()
        os.remove(self.path_fr)
        os.rename(self.b,self.path_fr)

        #指标提取
        coor_x,coor_y = -1, 0           # 初始值并无意义,只是先定义一下供后面的global赋值改变用于全局变量
        coor_m = np.array([[-1,-1]])
        coor = np.array([-1,[-1, -1]])
        
        


        

        def readVideo(cap,out,start_frameidx,end_frameidx,image1,image2,image3):
            global coor_x, coor_y,coor,coor_m
            # 检测是否正常打开：成功打开时，isOpened返回ture
            if cap.isOpened():
                for idx in range(start_frameidx,end_frameidx):
                    cap.set(cv2.CAP_PROP_POS_FRAMES,idx)
                    # 获取每一帧的图像frame
                    ret, frame = cap.read()
                    # 这里必须加上判断视频是否读取结束的判断,否则播放到最后一帧的时候出现问题了
                    if ret == True:
                        self.EditText.insert('end',"frame %s / %s"%(idx-start_frameidx,end_frameidx-start_frameidx))
                        self.EditText.insert(INSERT, '\n')
                        #start = time.time()
                        # 将从视频中获取的图像与要找的图像进行匹配

                        found = matchSift(image1,frame,MIN_MATCH_COUNT=30)
                        found = matchSift(image2,found,MIN_MATCH_COUNT=30)
                        found = matchSift(image3,found,MIN_MATCH_COUNT=20)
                        coor=np.row_stack((coor,[idx,coor_m]))
                        coor_m=[[-1,-1]]
                        out.write(found)
                    else:
                        break
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        break
            # 停止在最后的一帧图像上
            cv2.waitKey()
            out.release()
            # 如果任务完成了，就释放一切
            cap.release()
            # 关掉所有已打开的GUI窗口
            cv2.destroyAllWindows()

        def matchSift(findimg, img,MIN_MATCH_COUNT):
            global coor_m
            """转换成灰度图片"""
            gray1 = cv2.cvtColor(findimg, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            """创建SIFT对象"""
            sift = cv2.SIFT_create()
            """创建FLAN匹配器"""
            matcher = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})
            """检测关键点并计算键值描述符"""
            kpts1, descs1 = sift.detectAndCompute(gray1, None)
            kpts2, descs2 = sift.detectAndCompute(gray2, None)
            """KnnMatch获得Top2"""
            matches = matcher.knnMatch(descs1, descs2, 2)
            """根据他们的距离排序"""
            matches = sorted(matches, key=lambda x: x[0].distance)
            """比率测试，以获得良好的匹配"""
            good = [m1 for (m1, m2) in matches if m1.distance < 0.7 * m2.distance]

            canvas = img.copy()
            """发现单应矩阵"""
            """当有足够的健壮匹配点对（至少个MIN_MATCH_COUNT）时"""
            #MIN_MATCH_COUNT=110
            if len(good) >= MIN_MATCH_COUNT:
                #print(len(good))
                """从匹配中提取出对应点对"""
                """小对象的查询索引，场景的训练索引"""
                src_pts = np.float32([kpts1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kpts2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                """利用匹配点找到CV2.RANSAC中的单应矩阵"""
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                """计算图1的畸变，也就是在图2中的对应的位置"""
                h, w = findimg.shape[:2]
                pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                for i in range(dst.shape[0]):
                    coor_x=dst[i][0,0]
                    coor_y=dst[i][0,1]
                    coor_i=[coor_x,coor_y] 
                    coor_m=np.row_stack((coor_m,coor_i))

                """绘制边框"""
                cv2.polylines(canvas, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
            else:
                self.EditText.insert('end',"Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
                self.EditText.insert(INSERT, '\n')
                for i in range(4):
                    coor_x=-1
                    coor_y=-1
                    coor_i=[coor_x,coor_y]
                    coor_m=np.row_stack((coor_m,coor_i))
                return img
            return canvas

        def isInterArea(testPoint,AreaPoint):#testPoint为待测点[x,y]
            LTPoint = AreaPoint[0]#AreaPoint为按逆时针顺序的4个点[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            LBPoint = AreaPoint[1]
            RBPoint = AreaPoint[2]
            RTPoint = AreaPoint[3]

            a = (LTPoint[0]-LBPoint[0])*(testPoint[1]-LBPoint[1])-(LTPoint[1]-LBPoint[1])*(testPoint[0]-LBPoint[0])
            b = (RTPoint[0]-LTPoint[0])*(testPoint[1]-LTPoint[1])-(RTPoint[1]-LTPoint[1])*(testPoint[0]-LTPoint[0])
            c = (RBPoint[0]-RTPoint[0])*(testPoint[1]-RTPoint[1])-(RBPoint[1]-RTPoint[1])*(testPoint[0]-RTPoint[0])
            d = (LBPoint[0]-RBPoint[0])*(testPoint[1]-RBPoint[1])-(LBPoint[1]-RBPoint[1])*(testPoint[0]-RBPoint[0])
            #print(a,b,c,d)
            if (a>0 and b>0 and c>0 and d>0) or (a<0 and b<0 and c<0 and d<0):
                return True
            else:
                return False

        if __name__ == '__main__':
            # 视频的路径
            videopath = self.path_v

            # 读取视频，并识别
            fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
            ##创建并初始化VideoWriter对象，用于记录录屏视频，(1920,1080)取决于呈现刺激的屏幕分辨率
            #记录每个题目的编号、起始帧、结束帧
            stims_info = np.loadtxt(self.path_fr,delimiter=",")
            #print(stim_info.shape)

            with open(os.path.join(self.outputFilePath,'fixation_measure.csv'),"w") as f:
                for stim in stims_info:
                    stim_num = int(stim[0])
                    start_idx = int(stim[1])
                    end_idx = int(stim[2])
                    imagepath1 = os.path.join(self.path_p,'computer_area%d.jpg')%stim_num
                    #print(imagepath1)
                    imagepath2 = os.path.join(self.path_p,'interaction_area%d.jpg')%stim_num
                    imagepath3 = os.path.join(self.path_p,'ques_area%d.jpg')%stim_num
                    image1 = cv2.imread(imagepath1)
                    image2 = cv2.imread(imagepath2)
                    image3 = cv2.imread(imagepath3)
                    cap = cv2.VideoCapture(videopath)
                    out = cv2.VideoWriter(os.path.join(self.outputFilePath,'video_stim%d.mp4')%stim_num, fourcc, 30, (1280, 720))
                    #打开视频文件：在实例化的同时进行初始化
                    self.EditText.insert('end',"this is stim%d"%stim_num)
                    self.EditText.insert(INSERT, '\n')
                    readVideo(cap,out,start_idx,end_idx,image1,image2,image3)

                    f.write("number,%d\n"%(stim_num))
                    Ques_FixDuration = 0
                    Interaction_FixDuration = 0
                    Fix_meanDuration=0
                    Ques_FixCount = 0
                    Interaction_FixCount = 0
                    AOI_fix_list=[]
                    Interaction_FixDuration_list=[]
                    Ques_FixDuration_list=[]
                    for i in range(data3.shape[0]):
                        fix_duration = data3.iloc[i,2]
                        start_frame = data3.iloc[i,3]
                        end_frame = data3.iloc[i,4]
                        fix_x = data3.iloc[i,5]
                        fix_y = data3.iloc[i,6]
                        fix_i = [fix_x,fix_y]
                        Ques_frame = 0
                        Interaction_frame = 0
                        Total_frame=0
                        for j in range(len(coor)):
                            if start_frame<=coor[j][0]<=end_frame:
                                coor_n=coor[j][1]
                                AOI_computer = []
                                AOI_ques = []

                                for jj in range(1,5):
                                    AOI_computer.append(coor_n[jj])
                                for jj in range(9,13):
                                    AOI_ques.append(coor_n[jj])

                                if isInterArea(fix_i,AOI_computer):
                                    if isInterArea(fix_i,AOI_ques):
                                        Ques_frame+=1
                                        Total_frame+=1
                                    else:
                                        Interaction_frame+=1
                                        Total_frame+=1

                        if Total_frame>0:
                            if Ques_frame>=Total_frame*2/3:
                                Ques_FixCount+=1
                                Ques_FixDuration+=fix_duration
                                Ques_FixDuration_list.append(fix_duration)
                                AOI_fix_list.append(['Ques',fix_duration])
                            elif Interaction_frame>=Total_frame*2/3:
                                Interaction_FixCount+=1
                                Interaction_FixDuration+=fix_duration
                                Interaction_FixDuration_list.append(fix_duration)
                                AOI_fix_list.append(['Interaction',fix_duration])



                    if Interaction_FixCount !=0:
                        Interaction_MeanDuration = Interaction_FixDuration/Interaction_FixCount
                        Interaction_MaxFixDuration = np.max(Interaction_FixDuration_list)
                        Interaction_MinFixDuration = np.min(Interaction_FixDuration_list)
                        Interaction_firstFixDuration=Interaction_FixDuration_list[0]
                    else:
                        Interaction_FixDuration=0
                        Interaction_MeanDuration=0
                        Interaction_MaxFixDuration=0
                        Interaction_MinFixDuration=0
                        Interaction_firstFixDuration


                    if Ques_FixCount !=0:
                        Ques_MeanDuration = Ques_FixDuration/Ques_FixCount
                        Ques_MaxFixDuration = np.max(Ques_FixDuration_list)
                        Ques_MinFixDuration = np.min(Ques_FixDuration_list)
                        Ques_firstFixDuration=Ques_FixDuration_list[0]
                    else:
                        Ques_FixDuration=0
                        Ques_MeanDuration=0
                        Ques_MaxFixDuration=0
                        Ques_MinFixDuration=0
                        Ques_firstFixDuration=0

                    f.write("Interaction_TotalFixDuration,%f\n"%Interaction_FixDuration)
                    f.write("Ques_TotalFixDuration,%f\n"%Ques_FixDuration)
                    f.write("Interaction_MeanDuration,%f\n"%Interaction_MeanDuration)
                    f.write("Ques_MeanDuration,%f\n"%Ques_MeanDuration)
                    f.write("Interaction_firstFixDuration,%f\n"%Interaction_firstFixDuration)
                    f.write("Ques_firstFixDuration,%f\n"%Ques_firstFixDuration)
                    f.write("Interaction_MaxFixDuration,%f\n"%Interaction_MaxFixDuration)
                    f.write("Ques_MaxFixDuration,%f\n"%Ques_MaxFixDuration)
                    f.write("Interaction_MinFixDuration,%f\n"%Interaction_MinFixDuration)
                    f.write("Ques_MinFixDuration,%f\n"%Ques_MinFixDuration)
                    f.write("Interaction_TotalFixCount,%f\n"%Interaction_FixCount)
                    f.write("Ques_TotalFixCount,%f\n"%Ques_FixCount)

                    start=AOI_fix_list[0][0]
                    AOI_BeforeLeapDuration=0
                    leapcount=0#记录跳动次数

                    for i in range(len(AOI_fix_list)):
                        t=len(AOI_fix_list)

                        if AOI_fix_list[i][0]==start:
                            AOI_BeforeLeapDuration+=AOI_fix_list[i][1]
                            if i==t-1:
                                f.write("Duration on the AOI_%s,%f\n"%(AOI_fix_list[i][0],AOI_BeforeLeapDuration))
                        else:
                            leapcount+=1
                            f.write("Duration before moving from the AOI_%s to AOI_%s,%f\n"%(AOI_fix_list[i-1][0],AOI_fix_list[i][0],AOI_BeforeLeapDuration))    
                            start=AOI_fix_list[i][0]
                            AOI_BeforeLeapDuration=0
                            AOI_BeforeLeapDuration+=AOI_fix_list[i][1]

                    f.write("LeapCount,%d\n"%(leapcount))
                    coor = np.array([-1,[-1, -1]])
        self.EditText.insert('end',"Indicator_Extraction end")
    
    def RunPage(self):
        self.root1 = Tk()
        self.root1.title('Run Result')
        self.root1.geometry("600x450")
         
        t = threading.Thread(target=self.IE_Run)#将指标提取程序放入线程
        self.EditText = scrolledtext.ScrolledText(self.root1,width=71,height=23,font=("Arial", 12),fg='white',bg='black')
        self.EditText.place(x=1,y=1)
        t.setDaemon(True)#守护线程
        t.start()#开启线程   
        self.root1.mainloop()        

    #返回首页函数
    def bf_goFirst(self):
        #self.stop_thread(self.IE_thread)
        self.root.destroy()
        FirstPage()

    def __init__(self):
        self.root = Tk()
        self.root.geometry("600x450")
        self.root.title('Indicator Extraction')
        
        #变量path
        self.path_fix = StringVar()
        self.path_video = StringVar()
        self.path_pic = StringVar()
        self.path_frames = StringVar()
        self.outputpath = StringVar()
        self.filenewname = StringVar()
    
        #数据输入-一个frame
        Input = LabelFrame(self.root, text="Data Input",font=("Arial", 16), padx=15, pady=5)
        Input.grid(padx=18, pady=20)
        #注视点文件
        L1 = Label(Input, text="Fixations",font=("Arial", 14),fg='white',bg='tan1',width=14).grid(column=1,row=1,padx=5)
        E1 = Entry(Input, bd =5,textvariable = self.path_fix).grid(column=2,row=1,ipadx=65)
        button1 = Button(Input, text = "...",font=("Arial", 13),command = self.selectFile_fixations).grid(column=9,row=1,padx=5,ipadx=5)
        
        #视频
        L2 = Label(Input, text="Video",font=("Arial", 14),fg='white',bg='tan1',width=14).grid(column=1,row=3,padx=5)
        E2 = Entry(Input, bd =5,textvariable = self.path_video).grid(column=2,row=3,ipadx=65)
        button2 = Button(Input, text = "...",font=("Arial", 13),command=self.selectFile_video).grid(column=9,row=3,padx=5,ipadx=5)
        #匹配图片
        L3 = Label(Input, text="Match Pictures",font=("Arial", 14),fg='white',bg='tan1',width=14).grid(column=1,row=5,padx=5)
        E3 = Entry(Input, bd =5,textvariable = self.path_pic).grid(column=2,row=5,ipadx=65)
        button3 = Button(Input, text = "...",font=("Arial", 13),command=self.selectFile_matchPictures).grid(column=9,row=5,padx=5,ipadx=5)
        #起止帧数
        L4 = Label(Input, text="Start_End Frames",font=("Arial", 14),fg='white',bg='tan1',width=14).grid(column=1,row=7,padx=5)
        E4 = Entry(Input, bd =5,textvariable = self.path_frames).grid(column=2,row=7,ipadx=65)
        button4 = Button(Input, text = "...",font=("Arial", 13),command = self.selectFile_frames).grid(column=9,row=7,padx=5,ipadx=5)

        #数据输出-一个frame
        Output = LabelFrame(self.root, text="Data Output", font=("Arial", 16),padx=15, pady=5,width=200,height=50)
        Output.grid(padx=18, pady=5)
        L3 = Label(Output, text="Output Path",font=("Arial", 14),fg='white',bg='tan1',width=14).grid(column=1,row=8,padx=5)
        E3 = Entry(Output, bd =5,textvariable = self.outputpath).grid(column=2,row=8,ipadx=65)
        button3 = Button(Output, text = "...",font=("Arial", 13),command=self.outputFile).grid(column=9,row=8,padx=5,ipadx=5)

        #数据输出-一个frame
        Run = Frame(self.root,padx=15, pady=40,width=200,height=50)
        Run.grid(padx=18, pady=5)
        button4 = Button(Run, text = "Run",font=("Arial", 16),fg='white',bg='tan1',activebackground='tan2',                         width=15,command = self.RunPage).grid(column=1,row=60,padx=20,pady=15)
        button5 = Button(Run, text = "Go Back",font=("Arial", 16),fg='white',bg='tan1',activebackground='tan2',                         width=15,command=self.bf_goFirst).grid(column=20,row=60,padx=20,pady=15)

        self.root.mainloop()


if __name__ == '__main__':
    FirstPage()

