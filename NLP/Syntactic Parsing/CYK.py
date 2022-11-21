import nltk

def init_nfst(tok, gram): #normalize the terminal nodes of the tree
    numtokens1 = len(tok)  # 8 tokens
    print('numtokens1: ',numtokens1)
    nfst = [["." for i in range(numtokens1+1)] for j in range(numtokens1+1)]
    print(nfst)
    for i in range(numtokens1):
        print(i)
        prod = gram.productions(rhs=tok[i])
        print(prod)
        nfst[i][i + 1] = prod[0].lhs()
    return nfst

def display(nfst, tok): #display the result of parsing
    print("NFST" + ''.join([(" %4d"%i) for i in range(1, len(nfst))]))
    for i in range(0, len(nfst)-1): #(0,8)
        print("%6d"%i, end="  ")
        for j in range(1, len(nfst)):
            print("%-3s"%(nfst[i][j]),end='  ')
        print("")


def complete_nfst(nfst, tok): #combinate the terminal nodes with the rules stated
    global index1
    index1 = {}
    numtokens1 = len(tok)
    print(
        "%s %3s %s  %3s %s ==> %s %3s %s" % ("start", "nt1", "mid", "nt2", "end", "start", "index1[(nt1, nt2)]", "end"))
    print("-------------------------------------------------------------")
    for prod in gram.productions():
        index1[prod.rhs()] = prod.lhs()  # index = gram.productuon

    for span in range(2, numtokens1 + 1):  # range(2,9) > span = 2,3,4,5,6,7,8

            for start in range(numtokens1 + 1 - span):  # start > = 7,6,5,4,3,2,1
                end = start + span  

                for mid in range(start + 1, end):  # (8,9)mid: 8 / start 6 end 8 mid7:
                    nt1, nt2 = nfst[start][mid], nfst[mid][end]  
                    if (nt1, nt2) in index1:  
                        print("[%s]%6s [%s]%3s   [%s] ==> [%s]        %3s           [%s]" % (
                        start, nt1, mid, nt2, end, start, index1[(nt1, nt2)], end))
                        nfst[start][end] = index1[(nt1, nt2)]

    return nfst

"""
you can parse your sentence by editing the sentence and gram
"""
#nltk.download('punkt')
sentence = 'I eat sushi with chopsticks with you' #change the sentence if you want to
tok = nltk.word_tokenize(sentence)
print(len(tok))
gram = nltk.CFG.fromstring(""" 
S -> NP VP
NP -> NP PP
NP -> 'sushi'
NP -> 'I'
NP -> 'chopsticks'
NP -> 'you'
VP -> Verb NP  
VP -> VP PP
Verb -> 'eat'
PP -> Prep NP
Prep -> 'with'
""")

print(type(gram))

res1 = init_nfst(tok, gram)

res2 = complete_nfst(res1, tok)
print("\n")
display(res1, tok)
print("\n")
display(res2, tok)