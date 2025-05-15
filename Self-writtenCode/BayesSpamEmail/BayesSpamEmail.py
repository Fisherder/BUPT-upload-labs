# encoding=utf-8

import re
import jieba
import os


class SpamEmailBayes:
    # 获得停用词表(原代码)
    # def get_stop_words(self):
    #     stop_list = []
    #     for line in open(r'data\中文停用词表.txt',encoding="gbk"):
    #         stop_list.append(line[:len(line) - 1])
    #     return stop_list

    # 获得停用词表(修改后的代码)
    def get_stop_words(self):
        """Load stop words from file, handling file not found and encoding errors."""
        stop_list = []
        file_path = os.path.join('data', '中文停用词表.txt')
        try:
            with open(file_path, encoding='gbk') as file:
                for line in file:
                    stop_list.append(line.strip())
        except FileNotFoundError:
            print(f"Error: Stop words file '{file_path}' not found.")
            return stop_list
        except UnicodeDecodeError:
            print(f"Error: Failed to decode file '{file_path}' with gbk encoding.")
            return stop_list
        return stop_list

    # 获得词典
    def get_word_list(self, content, words_list, stop_list):
        # 分词结果放入res_list
        res_list = list(jieba.cut(content))
        for i in res_list:
            if i not in stop_list and i.strip() != '' and i != None:
                if i not in words_list:
                    words_list.append(i)

    # 若列表中的词已在词典中，则加1，否则添加进去
    def addToDict(self, words_list, words_dict):
        for item in words_list:
            if item in words_dict.keys():
                words_dict[item] += 1
            else:
                words_dict.setdefault(item, 1)

    def get_File_List(self, filePath):
        filenames = os.listdir(filePath)
        return filenames

    # 通过计算每个文件中p(s|w)来得到对分类影响最大的15个词(原代码)
    # def getTestWords(self, testDict, spamDict, normDict, normFilelen, spamFilelen):
    #     wordProbList = {}
    #     for word, num in testDict.items():
    #         if word in spamDict.keys() and word in normDict.keys():
    #             # 该文件中包含词个数
    #             pw_s = spamDict[word] / spamFilelen
    #             pw_n = normDict[word] / normFilelen
    #             ps_w = pw_s / (pw_s + pw_n)
    #             wordProbList.setdefault(word, ps_w)
    #         if word in spamDict.keys() and word not in normDict.keys():
    #             pw_s = spamDict[word] / spamFilelen
    #             pw_n = 0.01
    #             ps_w = pw_s / (pw_s + pw_n)
    #             wordProbList.setdefault(word, ps_w)
    #         if word not in spamDict.keys() and word in normDict.keys():
    #             pw_s = 0.01
    #             pw_n = normDict[word] / normFilelen
    #             ps_w = pw_s / (pw_s + pw_n)
    #             wordProbList.setdefault(word, ps_w)
    #         if word not in spamDict.keys() and word not in normDict.keys():
    #             # 若该词不在脏词词典中，概率设为0.4
    #             wordProbList.setdefault(word, 0.4)
    #     sorted(wordProbList.items(), key=lambda d: d[1], reverse=True)[0:15]
    #     return (wordProbList)

    # 通过计算每个文件中p(s|w)来得到对分类影响最大的15个词(修改后代码)
    def get_test_words(self, test_dict, spam_dict, norm_dict, norm_file_len, spam_file_len):
        """Get top 15 words with highest spam probability using Bayesian formula."""
        word_prob_list = {}
        for word, _ in test_dict.items():
            if word in spam_dict and word in norm_dict:
                pw_s = spam_dict[word] / spam_file_len
                pw_n = norm_dict[word] / norm_file_len
                ps_w = pw_s / (pw_s + pw_n)
                word_prob_list[word] = ps_w
            elif word in spam_dict and word not in norm_dict:
                pw_s = spam_dict[word] / spam_file_len
                pw_n = 0.01  # Small smoothing value for unseen normal words
                ps_w = pw_s / (pw_s + pw_n)
                word_prob_list[word] = ps_w
            elif word not in spam_dict and word in norm_dict:
                pw_s = 0.01  # Small smoothing value for unseen spam words
                pw_n = norm_dict[word] / norm_file_len
                ps_w = pw_s / (pw_s + pw_n)
                word_prob_list[word] = ps_w
            else:
                # Default probability for words not in spam or normal dictionaries
                word_prob_list[word] = 0.4
        # Return top 15 words sorted by spam probability
        sorted_probs = sorted(word_prob_list.items(), key=lambda d: d[1], reverse=True)[:15]
        return dict(sorted_probs)

    # 计算贝叶斯概率
    def calBayes(self, wordList, spamdict, normdict):
        ps_w = 1
        ps_n = 1

        for word, prob in wordList.items():
            print(word + "/" + str(prob))
            ps_w *= (prob)
            ps_n *= (1 - prob)
        p = ps_w / (ps_w + ps_n)
        #         print(str(ps_w)+"////"+str(ps_n))
        return p

        # 计算预测结果正确率

    def calAccuracy(self, testResult):
        rightCount = 0
        errorCount = 0
        for name, catagory in testResult.items():
            if (int(name) < 1000 and catagory == 0) or (int(name) > 1000 and catagory == 1):
                rightCount += 1
            else:
                errorCount += 1
        return rightCount / (rightCount + errorCount)


# spam类对象
spam = SpamEmailBayes()
# 保存词频的词典
spamDict = {}
normDict = {}
testDict = {}
# 保存每封邮件中出现的词
wordsList = []
wordsDict = {}
# 保存预测结果,key为文件名，值为预测类别
testResult = {}
# 分别获得正常邮件、垃圾邮件及测试文件名称列表
normFileList = spam.get_File_List('data/normal')
spamFileList = spam.get_File_List('data/spam')
testFileList = spam.get_File_List('data/test')
# 获取训练集中正常邮件与垃圾邮件的数量
normFilelen = len(normFileList)
spamFilelen = len(spamFileList)
# 获得停用词表，用于对停用词过滤
stopList = spam.get_stop_words()
# 获得正常邮件中的词频
for fileName in normFileList:
    wordsList.clear()
    for line in open('data/normal/' + fileName, encoding="gbk"):
        # 过滤掉非中文字符
        rule = re.compile(r"[^\u4e00-\u9fa5]")
        line = rule.sub("", line)
        # 将每封邮件出现的词保存在wordsList中
        spam.get_word_list(line, wordsList, stopList)
    # 统计每个词在所有邮件中出现的次数
    spam.addToDict(wordsList, wordsDict)
normDict = wordsDict.copy()

# 获得垃圾邮件中的词频
wordsDict.clear()
for fileName in spamFileList:
    wordsList.clear()
    for line in open('data/spam/' + fileName, encoding="gbk"):
        rule = re.compile(r"[^\u4e00-\u9fa5]")
        line = rule.sub("", line)
        spam.get_word_list(line, wordsList, stopList)
    spam.addToDict(wordsList, wordsDict)
spamDict = wordsDict.copy()

# 测试邮件
for fileName in testFileList:
    testDict.clear()
    wordsDict.clear()
    wordsList.clear()
    for line in open('data/test/' + fileName, encoding="gbk"):
        rule = re.compile(r"[^\u4e00-\u9fa5]")
        line = rule.sub("", line)
        spam.get_word_list(line, wordsList, stopList)
    spam.addToDict(wordsList, wordsDict)
    testDict = wordsDict.copy()
    # 通过计算每个文件中p(s|w)来得到对分类影响最大的15个词
    wordProbList = spam.getTestWords(testDict, spamDict, normDict, normFilelen, spamFilelen)
    # 对每封邮件得到的15个词计算贝叶斯概率
    p = spam.calBayes(wordProbList, spamDict, normDict)
    if (p > 0.9):
        testResult.setdefault(fileName, 1)
    else:
        testResult.setdefault(fileName, 0)
# 计算分类准确率（测试集中文件名低于1000的为正常邮件）
testAccuracy = spam.calAccuracy(testResult)
for i, ic in testResult.items():
    print(i + "/" + str(ic))

print("Accuracy:" + str(testAccuracy))

