import re
#from nltk.tokenize import *
from pprint import pprint
#import nltk.data
from nltk.tokenize import sent_tokenize
from itertools import groupby
import itertools
import nltk
from nltk.probability import *
from nltk.model.api import ModelI
#from nltk.corpus import brown
import sys, re, random, warnings
from math import *

# define alphabet (everything else will be deleted from corpus data)
# NB: "-" has to come last so the string can be used as a regexp character range below
alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'.!? -"

## --> choose your own corpora here (make sure to use similar texts, so that n-gram models work well)
corpus_root = '/Users/atandon/legal_corpus'
train_corpus = " ".join(nltk.corpus.PlaintextCorpusReader(corpus_root,'total_corpus.txt').words())
#train_corpus = " ".join(brown.words(categories="romance"))  # train models on romance
#devel_corpus = " ".join(brown.words(categories="humor"))    # optimise meta-parameters on humour
# NB: we cannot just use <corpus>.raw() for an annotated corpus!

## --> reduce corpus size if your computer isn't fast enough
# train_corpus = train_corpus[0:50000]
# devel_corpus = devel_corpus[0:50000]

## --> increase this value in order to experiment with higher-order models
n_max = 3

print "Training data:    %d characters" % (len(train_corpus))
#print "Development data: %d characters" % (len(devel_corpus))

# wrap corpus clean-up in function so it can easily be applied to all our corpora
def cleanup (corpus):
        corpus = re.sub(r'\s+', " ", corpus)       # replace linebreaks by blanks, normalise multiple blanks
        cleanup = re.compile("[^"+alphabet+"]+")   # this regular expression matches one or more invalid characters
        corpus = cleanup.sub("", corpus)           # delete all invalid characters (substitute with empty string)
        return corpus

#train_corpus = cleanup(train_corpus)
#devel_corpus = cleanup(devel_corpus)


def collect_frequency_data (corpus, n_max):
        fdata = [ None for i in range(n_max) ] # list of n-gram models for n-1 = 0 ... n_max-1
        size = len(corpus)
        for n1 in range(n_max):
                training_pairs = [
                        (corpus[i-n1:i], corpus[i])  # (history, next character) at position i in training corpus
                        for i in range(n1, size)     # where i ranges from n-1 to last index in corpus (Python uses 0-based indexing!)
                ]
                fdata[n1] = ConditionalFreqDist(training_pairs) # compile data into conditional frequency distribution
        return fdata

#train_fdata = collect_frequency_data(train_corpus, n_max)

class OurNgramModel(ModelI):
        def __init__ (self, fdata, n, q):
                self.n = n
                if (n > len(fdata)):
                        raise RuntimeError("Insufficient frequency data for %d-gram model." % (n))
                self.q = q
                if (q <= 0 or q >= 1):
                        raise RuntimeError("Interpolation parameter q=%f must be in range (0,1)." % (q))
                fdata = fdata[0:n] # determine conditional probabilities for required history sizes only
                self.cp = [
                        ConditionalProbDist(cfd, MLEProbDist)  ## --> change this line for add-lambda smoothing
                        for cfd in fdata
                ]
                # To be on the safe side, use add-one smoothing for the unigram probabilities instead of MLE
                # (the digit '7' does not occur in the romance texts of the Brown corpus, but 4 times in humour!)
                self.cp[0] = ConditionalProbDist(fdata[0], LaplaceProbDist, bins=len(alphabet))


        def prob (self, next_char, history):
                n1 = self.n - 1 # n1 = n - 1
                l = len(history)
                if (l < n1):
                        raise RuntimeError("History '%s' too short for %d-gram model." % (history, self.n))
                while (n1 > 0 and history[(l-n1):l] not in self.cp[n1]):
                        n1 -= 1 # fall back on shorter history if necessary (before interpolation / back-off !!)
                p = 0.0     # accumulate interpolated probability
                coef = 1.0  # geometric interpolation coefficient (unnormalised)
                denom = 0.0 # calculate normalising denominator by adding up unnormalised coefficients
                for k in range(n1, -1, -1):  # k = n-1, n-2, ..., 0
                        p += coef * self.cp[k][history[(l-k):l]].prob(next_char)
                        denom += coef
                        coef *= self.q
                return p / denom

        def logprob (self, next_char, history):
                return -log(self.prob(next_char, history), 2)

        def entropy (self, text):
                n1 = self.n - 1
                text = (" " * n1) + text # pad text with blanks for initial history
                H = 0.0
                for k in xrange(n1, len(text)):
                        H += self.logprob(text[k], text[(k-n1):k])
                return H

        # This is an additional method not specified by the ModelI API.  It validates the n-gram
        # model on a stretch of text, ensuring that (i) Pr(text) > 0 and (ii) that normalisation
        # constraints are satisfied by the model for all histories that occur in the text.
        def validate (self, text):
                n1 = self.n - 1
                text = (" " * n1) + text # pad text with blanks for initial history
                checked_histories = {}   # don't check the same history twice (validation is expensive!)
                for k in xrange(n1, len(text)):
                        history = text[(k-n1):k]
                        if self.prob(text[k], history) <= 0:
                                raise RuntimeError("Smoothing error: Pr(%s|%s) = 0" % (text[k], history))
                        if history not in checked_histories:
                                sum_p = 0.0
                                for c in alphabet:
                                        sum_p += self.prob(c, history)
                                if abs(sum_p - 1.0) >= .0001:  # allow some rounding error
                                        raise RuntimeError("Normalisation error: Sum Pr(*|%s) = %.6f (should be 1.0)" % (history, sum_p))
                                checked_histories[history] = 1



best_model = None
best_model1 = None
best_entropy = 999
#best_model1 = OurNgramModel(train_fdata, 3, 0.1)
#for n in range(1, n_max + 1):
#        model = OurNgramModel(train_fdata, 3, 0.1)
#        if n == 3:
#                best_model1 = model
#        train_entropy = model.entropy(train_corpus) / len(train_corpus)  # measure per-character entropy for easier comparison
#        devel_entropy = model.entropy(devel_corpus) / len(devel_corpus)
#        print "%d-gram model (q=0.1):  training text H = %4.2f bits/char  devel text H = %4.2f bits/char" % (n, train_entropy, devel_entropy)
#        if (devel_entropy < best_entropy):
#                best_model = model
#                best_entropy = devel_entropy

#print "Validating best model (%d-grams) with cross-entropy H = %.2f bits/char." % (best_model.n, best_entropy)
#best_model.validate(devel_corpus)




class PatternTree(object):
    def __init__(self):
        self.left = None
        self.right = None
        self.node = None
        self.relation=None
        
class RegexPattern(object):
    def __init__(self):
        self.pattern = None
        self.patternTree=None
        self.token_pattern = r'\d+|[A-Za-z]+|\w+|\<\<|\<|\$\+|\$|\(|\)|\&|\||\!' 
    
        
    def compile(self,pattern):
        self.pattern = pattern
        self.patternTree = self.buildTree(pattern)
        print self.patternTree.node
        return self.patternTree
    
        
    def searchTokens(self,parsed,ch):
        i = -1
        for token in parsed:
            if (token == ch and not isinstance(token,PatternTree)):
                #print "found at:" + str((i+1))
                return i + 1
            i = i + 1
        return -1
    
    
    
    def buildTree(self,pattern):
        print "Got Request : " + pattern
        # extract root node
        items = re.findall(self.token_pattern, pattern)
        #pprint (items)
        root = PatternTree()

            
        parsed = []
        pcount = 0
        expr = ""
        for i in range(0,len(items)):
            
            if items[i] == '(':
                pcount = pcount +1
            if items[i] == ')':
                pcount = pcount - 1
                if pcount == 0:
                    parsed.append(self.buildTree(expr))
                    expr=""
            if pcount > 0 and not (items[i] == '(' and pcount==1):
                expr = expr + items[i]
            if pcount == 0 and items[i]!=')':
                parsed.append(items[i])
                
        
        
        #print "Parse len " + str(len(parsed))
        if len(parsed) == 2 and isinstance(parsed[1],PatternTree) and not isinstance(parsed[0],PatternTree) and (parsed[1].node == "&" or parsed[1].node == "|"):
            root = PatternTree()
            root.node=parsed[0]
            root.left=parsed[1]
            return root
        
        if len(parsed) == 3:
            node = PatternTree()
            root.node = parsed[0]
            if isinstance(parsed[2], PatternTree):
                node = parsed[2]
            else:
                node.node = parsed[2]
            node.relation = parsed[1]
            root.left = node
            return root
        if len(parsed) > 3:
            while(True):
                print "problem1 " + str(len(parsed))
                pprint(parsed)
                if len(parsed) == 1:
                    return parsed[0]
                if len(parsed) == 3:
                    node = PatternTree()
                    root.node = parsed[0]
                    node.node = parsed[2]
                    node.relation = parsed[1]
                    root.left = node
                    return root
                if len(parsed) == 2 and isinstance(parsed[1],PatternTree) and not isinstance(parsed[0],PatternTree) and (parsed[1].node == "&" or parsed[1].node == "|"):
                    print "COMING ABOOOOOVE"
                    root = PatternTree()
                    root.node=parsed[0]
                    root.left=parsed[1]
                    return root
                while (True):
                    if len(parsed) <= 3:
                        break
                    
                    pos = self.searchTokens(parsed,"&")
                    
                    if pos == -1 or pos == 0 or pos ==1:
                        break
                    ro = PatternTree()
                    ro.node="&"
                    
                    if isinstance(parsed[pos+1],PatternTree) and isinstance(parsed[pos-1],PatternTree) :
                        node = parsed[pos+1]
                        ro.right = node
                        parsed[pos] = ro
                        parsed.pop(pos+1)
                        node = parsed[pos-1]
                        ro.left = node
                        parsed[pos] = ro
                        parsed.pop(pos-1)
                        continue
                    
                    pprint(parsed)
                    if isinstance(parsed[pos+2], PatternTree):
                        print "VVVVVVVVVVV"
                        node = parsed[pos+2]
                        ro.right = node
                        node.relation = parsed[pos+1]
                        parsed.pop(pos+2)
                        parsed.pop(pos+1)
                        parsed[pos] = ro
                    else:
                        node = PatternTree()
                        node.relation = parsed[pos+1]
                        node.node = parsed[pos+2]
                        ro.right = node
                        parsed.pop(pos+2)
                        parsed.pop(pos+1)
                        parsed[pos] = ro
                    pprint(parsed)    
                    if isinstance(parsed[pos-1], PatternTree):
                        node = parsed[pos-1]
                        ro.left = node
                        node.relation = parsed[pos-2]
                        parsed.pop(pos-1)
                        parsed.pop(pos-2)
                        parsed[pos-2] = ro
                    else:
                        node = PatternTree()
                        node.relation = parsed[pos-2]
                        node.node = parsed[pos-1]
                        ro.left = node
                        parsed.pop(pos-1)
                        parsed.pop(pos-2)
                        parsed[pos-2] = ro
                    print "POSTY LEFT:" + str(ro.left) + " RIGHT:" + str(ro.right)     
                while (True):
                    print "problem2"
                    pos = self.searchTokens(parsed,"|")
                    
                    if pos == -1 or pos == 0 or pos ==1:
                        break
                    ro = PatternTree()
                    ro.node="|"
                    
                    #pprint(parsed)
                    if isinstance(parsed[pos+1],PatternTree) and isinstance(parsed[pos-1],PatternTree) :
                        node = parsed[pos+1]
                        ro.right = node
                        parsed[pos] = ro
                        parsed.pop(pos+1)
                        node = parsed[pos-1]
                        ro.left = node
                        parsed[pos] = ro
                        parsed.pop(pos-1)
                        continue
                        
                    if isinstance(parsed[pos+2], PatternTree):
                        node = parsed[pos+2]
                        ro.right = node
                        node.relation = parsed[pos+1]
                        parsed.pop(pos+2)
                        parsed.pop(pos+1)
                        parsed[pos] = ro
                    else:
                        node = PatternTree()
                        node.relation = parsed[pos+1]
                        node.node = parsed[pos+2]
                        ro.right = node
                        parsed.pop(pos+2)
                        parsed.pop(pos+1)
                        parsed[pos] = ro
                        
                    if isinstance(parsed[pos-1], PatternTree):
                        node = parsed[pos-1]
                        ro.left = node
                        node.relation = parsed[pos-2]
                        parsed.pop(pos-1)
                        parsed.pop(pos-2)
                        parsed[pos-2] = ro
                    else:
                        node = PatternTree()
                        node.relation = parsed[pos-2]
                        node.node = parsed[pos-1]
                        ro.left = node
                        parsed.pop(pos-1)
                        parsed.pop(pos-2)
                        parsed[pos-2] = ro        
                        
    

class PatternMatcher(object):
    def __init__(self,pattern):
        self.pattern= pattern
       
    def find(self):
        self.print1()
        return None
    
    def print1(self):
        print ""
        #print(self.pattern.patternTree.node)
        


class PatternList(object):
    def __init__(self,search,fro,to):
        self.search = RegexPattern().compile(search)    
        self.replaceFrom = RegexPattern().compile(fro)
        self.replaceTo = RegexPattern().compile(to)
        


class PatternSubstitutor(object):
    def __init__(self):
        self.fileName = "/Users/atandon/Desktop/patterns.txt"
        self.patterns = []
        self.foundPatterns = []
        self.duplicateNode = []
        
        f = open(self.fileName, 'r')
        slist = []
        for line in f:
            print "read:" + line
            pats = line.split('|||')          
            self.patterns.append(PatternList(pats[0],pats[1],pats[2]))
            print " WORKING >>>"
            
    def process(self,tree):
        print " Check1"
        for pattern in self.patterns:
             
            pprint(pattern.search)
            self.applyPattern(tree,pattern)
        return tree    
            
    def seachNode(self,node):
        for nodes in self.duplicateNode:
            if nodes == node:
                print "******Found duplicate*************"
                return False
        return True    
            
    def applyPattern(self,tree,pattern):
        #search = pattern.search.node
        print "Check2" + str(pattern.search.node)
        self.traverse(tree,pattern,None,0)
        
    def testCondition(self,patternTree,tree,parent,cID):
        if patternTree.relation == '<':
            print "looking for a child"
            
            flag = 0
            childCount = -1
            try:
                tree.node
            except AttributeError:
                if tree[0] == patternTree.node:
                    print " Test success******" + patternTree.node
                    if (self.seachNode(tree)):
                        self.foundPatterns.append(CustomTreeNode(tree,parent,cID))
                        self.duplicateNode.append(tree)
                    return True
            for child in tree:
                try:
                    child.node
                except AttributeError:
                    nn = ""
                    if isinstance(child,str):
                        nn = child
                    else:
                        nn = child[1]
                    if nn == patternTree.node:
                        
                        if (self.seachNode(child)):
                            self.foundPatterns.append(child)
                            self.duplicateNode.append(child)
                        nextNodeID = childCount
                            
                        parentNode = tree
                        nextNode = child
                        flag = 1    
                else:
                    if child.node == patternTree.node:
                      
                        if (self.seachNode(child)):
                            self.foundPatterns.append(CustomTreeNode(child,parent,childCount))
                            self.duplicateNode.append(child)
                        parentNode = tree
                        nextNode = child
                        nextNodeID = childCount
                        flag = 1

            # if sister not found return False            
            if flag == 0:
                return False
            
        if patternTree.relation == "$":
            print "Looking for a sister"
            flag = 0
            childCount = -1
            
            
            for child in parent:
                childCount = childCount  + 1
                
                try:
                    child.node
                except AttributeError:
                    if child[1] == patternTree.node:
                        print "^^^^^^^Matched :" + child[1] + " with " + patternTree.node
                        if (self.seachNode(child)):
                           self.duplicateNode.append(child)
                        
                           self.foundPatterns.append(CustomTreeNode(child,parent,childCount))
                        parentNode = parent
                        nextNode = child
                        nextNodeID = childCount
                        flag = 1
                else:
                    if child.node == patternTree.node:
                        if (self.seachNode(child)):
                           self.duplicateNode.append(child)
                           self.foundPatterns.append(child)
                        parentNode = parent
                        nextNode = child
                           
                        nextNodeID = childCount
                        flag = 1
            # if sister not found return False            
            if flag == 0:
                return False
        flag = 1
        if patternTree.left:
            if self.testCondition(patternTree.left,nextNode,parentNode,nextNodeID) == True:
                flag =1
            else:
                flag =0
        if patternTree.right:
            if self.testCondition(patternTree.right,nextNode,parentNode,nextNodeID) == True:
                flag = 1
            else:
                flag = 0
 
        if flag == 1:
            return True
        else:
            return False
            
            
        
    def specificSearch(self,tree,parent,cID,pattern):
        if pattern.node != tree[1]:
            return False
        flag = 0
        if pattern.left:
            if self.testCondition(pattern.left,tree,parent,cID) == True: 
                flag = 1
            else:
                flag = 0
        if pattern.right:
            if self.testCondition(pattern.right,tree,parent,cID) == True:
                flag = 1
            else:
                flag = 0
        if flag == 1:
            return True
        else:
            return False
        
    
    def traverse(self,t,pattern,parent,cID):
        
        try:
           t.node
        except AttributeError:
           if t[1] ==  pattern.search.node:
                
               if self.specificSearch(t,parent,cID,pattern.search):
                   print "****Found****"
                   
                   for t1 in [key for key,_ in groupby(self.foundPatterns)]:
                       pprint (t1)
                       if (not isinstance(t1,tuple)):
                           print "IIInstace found"
                           if self.specificSearch(t1.child,t1.parent,t1.id,pattern.replaceFrom):
                               print "WOWWWWW"
                               self.substitute(t1.child,t1.parent,pattern.replaceTo,t1.id)
                            #print t[1]
                   self.foundPatterns = []
                   self.duplicateNode = []
           print "**" , t[0],
        else:
           # Now we know that t.node is defined
           print '($$', t.node,
           i = 0
           for child in t: 
               self.traverse(child,pattern,t,i)
               i = i + 1
           print ')',    

    def substitute(self,tree,parent,patternTree,id1):
        # Create a new pattern tree out of patternTree
        print "Test111111"
        newTree = self.constructTree(patternTree)
        #newTree.draw()
        print "At ID " + str(id1)
        parent.draw()
        parent.__delitem__(id1)
        #parent.__setitem__(id1,newTree)
        parent.append(newTree)
        parent.draw()
        
        
        # substitute tree with the new Tree
        
    def constructTree(self,patternTree):
        print "Pattern Tree " + patternTree.node
        if patternTree == None:
            return
        if patternTree.right == None and patternTree.left!= None and patternTree.relation == "<" and patternTree.left.left == None and patternTree.left.right == None:
            print "forming  Tree for " + patternTree.node + "<" + patternTree.left.node
            return nltk.Tree(patternTree.node,[patternTree.left.node])
        if patternTree.node == "&":
            print "LEFT::" + str(patternTree.left) + " RIGHT:" + str(patternTree.right)
        if patternTree.node == "&" and patternTree.left != None and patternTree.right!= None:
            print "Found Something"
            return nltk.Tree("",[self.constructTree(patternTree.left),self.constructTree(patternTree.right)])
        if patternTree.left != None and patternTree.right == None:
            return nltk.Tree(patternTree.node,[self.constructTree(patternTree.left)])
        if patternTree.right != None and patternTree.left == None:
            return nltk.Tree(patternTree.node,[self.constructTree(patternTree.right)])    
 
class CustomTreeNode(object):
    def __init__(self,child,parent,id1):
        self.child = child
        self.parent = parent
        self.id = id1
                
class TreeRegexSubstitute(object):
    def __init__(self):
        grammar = r"""
                  NP: {<DT|JJ|NN.*>+}          # Chunk sequences of DT, JJ, NN
                  PP: {<IN><NP>}               # Chunk prepositions followed by NP
                  VP: {<VB.*><NP|PP|CLAUSE>+$} # Chunk verbs and their arguments
                  CLAUSE: {<NP><VP>}           # Chunk NP, VP
                  """
        self.cp = nltk.RegexpParser(grammar)
        self.patternSubstitute = PatternSubstitutor()
        

        
    def process(self,slist):
        output = []
        
        for sent in slist:
            for s in sent:
                #pprint (s)
                words = nltk.word_tokenize(s)
                pprint(words)
                tokens = nltk.pos_tag(words) 
                pprint (tokens)
                tr = self.cp.parse(tokens)
                pprint(tr)
                tr.draw()
                tr = self.patternSubstitute.process(tr)
                trans = tr.flatten().leaves()
                list1 = []
                for tr in trans:
                    if (isinstance(tr,tuple)):
                        if tr[0] != ".":
                           list1.append(tr[0])
                    else:
                        if tr != ".":
                           list1.append(tr)
                print " Done"
                finalSentence = list1[0]
                maxscore = 0
                #for sent in list(itertools.permutations(list1)):
                #    sentence = sent.join(" ")
                #    if (best_model1.entropy(sentence)/len(sentence)) > maxscore:
                #        maxscore = (best_model1.entropy(sentence)/len(sentence))
                #        finalSentence = sentence
                #print "Done1:::" + finalSentence
                
        return output
    






class LawConverter(object):
    def __init__(self,fileName):
        self.fileName = fileName
        self.lawSub = TreeRegexSubstitute()
        
    def process(self):
        f = open(self.fileName, 'r')
        slist = []
        for line in f:
            print "read:" + line
            slist.append(sent_tokenize(line))
            pprint (self.lawSub.process(slist))
            slist = []
            
        





def main():

    lawConvert = LawConverter("/Users/atandon/Desktop/input.txt")
    lawConvert.process()    



if __name__ == '__main__':
    main() 