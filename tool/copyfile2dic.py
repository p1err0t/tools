import os, re, shutil
# 移动文件的MoveFile类
class MoveFile(object):
    def __init__(self, srcDir, dstDir, recursive=True, flag='.DOC'):
        self.srcDir = srcDir # 源目录
        self.dstDir = dstDir # 目标目录
        self.recursive = recursive # 递归查找文件夹，默认为False，只查找源目录下的文件，不会递归查找其子文件夹
        self.flag = flag # 要匹配的文件类型 '.DOC'-doc文件
        self.duplicateFileName = [] # 捕获重复移动文件，即目标文件中已存在此文件
        self.badFileName = [] # 文件类型符合要求，但命名不符合要求
        self.docFile = [] # 捕获的jpg文件
        self.srcDirDict = {} # 当递归查找时，记录文件的root目录，在移动时使用
    # 找到给定类型（flag）的文件
    def findAllDOC(self):
        # recursively find file 
        if self.recursive == False:        
            for item in os.listdir(self.srcDir):
                if os.path.isfile(os.path.join(self.srcDir,item)) and \
                                os.path.splitext(item)[-1] == self.flag.lower():
                    self.docFile.append(item)
        else:
            for root, dirs, files in os.walk(self.srcDir):
                for item in files:
                    if os.path.splitext(item)[-1] == self.flag.lower():
                        self.docFile.append(item)
                        print(root)
                        self.srcDirDict[item] = root

        if not self.docFile:
            print('NOT FIND ANY DOC FILE!')
        return self.docFile
    # 使用正则表达式匹配
    # def parse(self, text):
    #     try:
    #         pat =re.compile('[\s\S]*')
    #         match = pat.match(text)
    #         data = match.group(1)
    #         fileName = data[:4]+'-'+data[4:6]
    #     except:
    #         self.badFileName.append(text)
    #         fileName = None
    #     return fileName
    # 移动
    def move(self, text):
        try:
            fileName = text
            # if fileName == None: return
            # if not os.path.isdir(os.path.join(self.dstDir, fileName)):# 判断主目录是否存在子文件夹
            #     os.mkdir(os.path.join(self.dstDir,fileName))

            srcPath= os.path.join(self.srcDirDict[text], text)
            dstDir = os.path.join(self.dstDir, fileName)
            shutil.copy(srcPath, dstDir)
        except:
            self.duplicateFileName.append(text)
            raise

    @staticmethod
    def decC(dir):
        return os.path.join(self.srcDir, dir)
    # 运行
    def run(self):
        try:
            if not os.path.isdir(self.dstDir):
                os.mkdir(self.dstDir)
            for text in self.findAllDOC():
                print(text)
                self.move(text)
            print('MOVE SUCCESSFUL!') 
        except TypeError:
            raise
# 例子： 将srcDir中的符合parse要求的doc文件移动到dstDir
srcDir = r'D:\wuhan'
dstDir = r'D:\wuhandoc'
fmv = MoveFile(srcDir, dstDir, recursive = True)
fmv.run()
