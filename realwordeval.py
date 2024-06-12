import nltk
anspath='./realwordans.txt'
resultpath='./realwordresult.txt'
ansfile=open(anspath,'r')
resultfile=open(resultpath,'r')
count=0
for i in range(3):
    ansline=ansfile.readline().split('\t')[1]
    ansset=set(nltk.word_tokenize(ansline))
    resultline=resultfile.readline().split('\t')[1]
    resultset=set(nltk.word_tokenize(resultline))
    if ansset==resultset:
        count+=1
    else:
        print(i)
        print(ansset)
        print(resultset)
        print('DIFF:',ansset-resultset,resultset-ansset)
        print()
print("Accuracy is : %.2f%%" % (count*1.00))
