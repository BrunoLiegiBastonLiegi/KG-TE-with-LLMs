# import warnings filter
from warnings import simplefilter
from sklearn.exceptions import UndefinedMetricWarning
# ignore all UndefinedMetricWarning warnings
simplefilter(action='ignore', category=UndefinedMetricWarning)
from bs4 import BeautifulSoup
import os, sys
import regex as re
import itertools
import statistics
import sys
from nervaluate import Evaluator
import nltk
from nltk.util import ngrams
import string
from sklearn.metrics import precision_score, recall_score, f1_score, multilabel_confusion_matrix
from sklearn import preprocessing
from tqdm import tqdm

sys.path.append('../../')
from utils import normalize_triple

currentpath = os.getcwd()

def getRefs(filepath):
    print('> Collecting references...')
    with open(filepath, encoding='utf-8') as fp:
        refssoup = BeautifulSoup(fp, 'lxml')

    refsentries = refssoup.find('benchmark').find('entries').find_all('entry')

    allreftriples = []

    for entry in tqdm(refsentries):
        entryreftriples = []
        try:
            modtriplesref = entry.find('modifiedtripleset').find_all('mtriple')
        except:
            modtriplesref = entry.find('generatedtripleset').find_all('gtriple')
        for modtriple in modtriplesref:
            entryreftriples.append(normalize_triple(modtriple.text))
        allreftriples.append(entryreftriples)

    newreflist = []

    print('> Normalizing references...')
    for entry in tqdm(allreftriples):
        newtriples = []
        for triple in entry:
            newtriple = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", triple).lower()
            newtriple = re.sub(r'_', ' ', newtriple).lower()
            newtriple = re.sub(r'\s+', ' ', newtriple).lower()
            adjusttriple = newtriple.split(' | ')
            manualmodified = re.search(r'^(.*?)(\s\((.*?)\))$', adjusttriple[-1])
            if manualmodified:
                adjusttriple[-1] = manualmodified.group(1)
                newtriple = ' | '.join(adjusttriple)
            newtriples.append(newtriple)
        newreflist.append(newtriples)

    return allreftriples, newreflist

def getCands(filepath):
    print('> Collecting candidates...')
    with open(filepath, encoding='utf-8') as fp:
        candssoup = BeautifulSoup(fp, 'lxml')

    candssentries = candssoup.find('benchmark').find('entries').find_all('entry')

    allcandtriples = []

    for entry in tqdm(candssentries):
        entrycandtriples = []
        modtriplescand = entry.find('generatedtripleset').find_all('gtriple')
        for modtriple in modtriplescand:
            entrycandtriples.append(normalize_triple(modtriple.text))
        allcandtriples.append(entrycandtriples)

    newcandlist = []

    print('> Normalizing candidates...')
    for entry in tqdm(allcandtriples):
        newtriples = []
        for triple in entry:
            newtriple = re.sub(r"([a-z])([A-Z])", "\g<1> \g<2>", triple).lower()
            newtriple = re.sub(r'_', ' ', newtriple).lower()
            newtriple = re.sub(r'\s+', ' ', newtriple).lower()
            adjusttriple = newtriple.split(' | ')
            manualmodified = re.search(r'^(.*?)(\s\((.*?)\))$', adjusttriple[-1])
            if manualmodified:
                adjusttriple[-1] = manualmodified.group(1)
                newtriple = ' | '.join(adjusttriple)
            newtriples.append(newtriple)
        newcandlist.append(newtriples)

    return allcandtriples, newcandlist

def find_sub_list(sl,l):
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            return ind,ind+sll-1

#We are going to try to find matches with the reference, starting with the highest chunk possible (all the words in the reference).
#If we don't find that, we are going to search for all n-grams -1 the number of words in the reference; than -2; than -3; etc.
def nonrefwords(newreflist, newcandlist, foundnum, ngramlength):
    while ngramlength > 0:
        #Get a list of all the ngrams of that size
        ngramlist = list(ngrams(newcandlist, ngramlength))
        for ngram in ngramlist:
            #If we find this ngram (in the same order) in the reference
            if find_sub_list(list(ngram), newreflist) is not None:
                #We're getting the start and end index of the ngram in the reference
                findnewref = find_sub_list(list(ngram), newreflist)
                #And all the numbers in between
                newrefindex = list(range(findnewref[0], findnewref[1] + 1))
                #Change the matched words to FOUNDREF-[FOUNDNUMBER]-[FOUNDINDEX]
                for idx in newrefindex:
                    newreflist[idx] = 'FOUNDREF-' + str(foundnum) + '-' + str(idx)

                #Now find the start and end index of the ngram in the candidate as well
                findnewcand = find_sub_list(list(ngram), newcandlist)
                #And all the indices in between
                newcandindex = list(range(findnewcand[0], findnewcand[1]+1))
                # Change the matched words to FOUNDCAND-[FOUNDNUMBER]-[REFERENCE-FOUNDINDEX]
                for idx, val in enumerate(newcandindex):
                    newcandlist[val] = 'FOUNDCAND-' + str(foundnum) + '-' + str(newrefindex[idx])
                foundnum += 1
                #And try to find new matches again
                nonrefwords(newreflist, newcandlist, foundnum, ngramlength)
        #If no match is found, try to find matches for ngrams 1 smaller
        ngramlength -= 1
    #Return the new lists if all possible ngrams have been searched
    return newreflist, newcandlist

def getrefdict(newreflist, newcandlist, tripletyperef, tripletypecand, baseidx):
    try:
        #If some match is found with the reference
        firstfoundidx = newcandlist.index([i for i in newcandlist if re.findall(r'^FOUNDCAND', i)][0])
        candidatefound = 'y'
    except IndexError:
        candidatefound = 'n'

    if candidatefound == 'y':
        unlinkedlist = []
        beforelist = []
        afterlist = []

        #If the first found candidate match is also the first word in the reference
        if newcandlist[firstfoundidx].endswith('-0'):
            #Flag that some words can appear before the first match, and they are linked with the first candidate match
            beforelinked = 'y'
            firstcand = re.search(r'^(FOUNDCAND-\d+)-', newcandlist[firstfoundidx]).group(1)
        else:
            beforelinked = 'n'

        lastfoundidx = None
        afterlinked = None
        #If there's more words after the last reference, link those to the last reference as well
        #If the last reference word is linked, but the last candidate word is not, one criterion of linking the last words is met
        if (newreflist[-1].startswith('FOUNDREF')) and (not newcandlist[-1].startswith('FOUNDCAND')):
            #If the last linked reference word is the last linked candidate word, the other criterion is also met.
            lastfound = [i for i in newcandlist if re.findall(r'^FOUNDCAND', i)][-1]
            candversion = newreflist[-1].replace('FOUNDREF', 'FOUNDCAND')
            if lastfound == candversion:
                lastfoundidx = newcandlist.index([i for i in newcandlist if re.findall(r'^FOUNDCAND', i)][-1])
                afterlinked = 'y'
                lastcand = re.search(r'^(FOUNDCAND-\d+)-', lastfound).group(1)


        #Ensure that all the not-found blocks are separated by giving them different unlinknumbers
        unlinknumber = 1
        for idx, can in enumerate(newcandlist):
            if not can.startswith('FOUNDCAND'):
                if (idx < firstfoundidx) and (beforelinked == 'y'):
                    newcandlist[idx] = firstcand + '-LINKED'
                    beforelist.append(firstcand + '-LINKED')
                elif (lastfoundidx != None) and (afterlinked != None) and (idx > lastfoundidx) and (afterlinked == 'y'):
                    newcandlist[idx] = lastcand + '-LINKED'
                    afterlist.append(lastcand + '-LINKED')
                else:
                    unlinkedlist.append('NOTFOUND-' + str(unlinknumber))
            else:
                unlinknumber += 1

        totallist = beforelist + newreflist + afterlist + unlinkedlist

        refstart = len(beforelist)
        refend = (len(beforelist) + len(newreflist)) - 1

        refdictlist = [{'label': tripletyperef, 'start': baseidx + refstart, 'end': baseidx + refend}]

        totallist2 = [x.replace('FOUNDREF', 'FOUNDCAND') for x in totallist]

        canddictlist = []
        currentcandidate = ''
        beginidx = ''
        endidx = ''
        collecting = 'n'
        for idx, candidate in enumerate(totallist2):
            if (candidate.startswith('FOUNDCAND')) or (candidate.startswith('NOTFOUND')):
                collecting = 'y'
                curcan = re.search(r'^((.*?)-\d+)', candidate).group(1)
                if curcan != currentcandidate:
                    if currentcandidate != '':
                        endidx = idx-1
                        canddictlist.append({'label': tripletypecand, 'start': baseidx + beginidx, 'end': baseidx + endidx})
                    currentcandidate = curcan
                    beginidx = idx

                if idx == len(totallist2)-1:
                    endidx = idx
                    canddictlist.append({'label': tripletypecand, 'start': baseidx + beginidx, 'end': baseidx + endidx})
            else:
                if collecting == 'y':
                    endidx = idx-1
                    canddictlist.append({'label': tripletypecand, 'start': baseidx + beginidx, 'end': baseidx + endidx})

    else:
        if len(newreflist) == 0:
            refdictlist = []
            canddictlist = [{'label': tripletypecand, 'start': baseidx, 'end': baseidx + (len(newcandlist) - 1)}]
            totallist = newcandlist
        elif len(newcandlist) == 0:
            canddictlist = []
            refdictlist = [{'label': tripletyperef, 'start': baseidx, 'end': baseidx + (len(newreflist) - 1)}]
            totallist = refdictlist
        else:
            totallist = newreflist + newcandlist
            refdictlist = [{'label': tripletyperef, 'start': baseidx, 'end': baseidx + (len(newreflist) - 1)}]
            canddictlist = [{'label': tripletypecand, 'start': baseidx + len(newreflist), 'end': baseidx + (len(totallist) - 1)}]


    return candidatefound, refdictlist, canddictlist, totallist

def evaluaterefcand(reference, candidate):
    newreference = reference.split(' | ')
    newcandidate = candidate.split(' | ')

    #Make sure that reference or candidate aren't '' values originally.
    if (len(newreference) > 1) and (len(newcandidate) > 1):
        indextriple = newreference
    elif (len(newreference) == 1) :
        indextriple = newcandidate
        newreference = ['', '', '']
    else:
        indextriple = newreference
        newcandidate = ['', '', '']

    subjectreflist = None
    subjectcandlist = None
    subjecttotallist = None
    predicatereflist = None
    predicatecandlist = None
    predicatetotallist = None
    objectreflist = None
    objectcandlist = None
    objecttotallist = None
    subjectfound = ''
    predicatefound = ''
    objectfound = ''

    for idx, attrib in enumerate(indextriple):
        #Let's go over each attribute of the triple one by one
        refsub = newreference[idx]
        candsub = newcandidate[idx]

        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)

        reflist = [x.lower() for x in reflist if re.search(r'^[' + re.escape(string.punctuation) + r']+$', x) == None]
        candlist = [x.lower() for x in candlist if re.search(r'^[' + re.escape(string.punctuation) + r']$', x) == None]

        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        #Start with an ngram the full number of words in the reference
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)
        if idx == 0:
            candidatefound, refdictlist, canddictlist, totallist = getrefdict(newreflist, newcandlist, 'SUB', 'SUB', 0)
            subjectfound = candidatefound
            subjectreflist = refdictlist.copy()
            subjectcandlist = canddictlist.copy()
            subjecttotallist = totallist.copy()
        elif idx == 1:
            candidatefound, refdictlist, canddictlist, totallist = getrefdict(newreflist, newcandlist, 'PRED', 'PRED', len(subjecttotallist))
            predicatefound = candidatefound
            predicatereflist = refdictlist.copy()
            predicatecandlist = canddictlist.copy()
            predicatetotallist = totallist.copy()
        else:
            candidatefound, refdictlist, canddictlist, totallist = getrefdict(newreflist, newcandlist, 'OBJ', 'OBJ', len(subjecttotallist) + len(predicatetotallist))
            objectfound = candidatefound
            objectreflist = refdictlist.copy()
            objectcandlist = canddictlist.copy()
            objecttotallist = totallist.copy()

    switchmatchfound = 'n'
    #If no matches were found for two or more attributes, we are going to try and compare different attributes to each other.
    #First let's try to match the candidate subject and reference object (and vice versa)
    if (subjectfound == 'n') and (objectfound == 'n'):
        refsub = newreference[0]
        candsub = newcandidate[2]

        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)

        reflist = [x.lower() for x in reflist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]
        candlist = [x.lower() for x in candlist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]

        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        # Start with an ngram the full number of words in the candidate
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)

        candidatefound, refdictlist, canddictlist, totallist = getrefdict(newreflist, newcandlist, 'SUB', 'OBJ', 0)

        refsub = newreference[2]
        candsub = newcandidate[0]

        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)

        reflist = [x.lower() for x in reflist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]
        candlist = [x.lower() for x in candlist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]

        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        # Start with an ngram the full number of words in the candidate
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)
        candidatefound2, refdictlist2, canddictlist2, totallist2 = getrefdict(newreflist, newcandlist, 'OBJ', 'SUB', len(totallist) + len(predicatetotallist))

        if (candidatefound == 'y') or (candidatefound2 == 'y'):
            subjectfound = candidatefound
            subjectreflist = refdictlist.copy()
            subjectcandlist = canddictlist.copy()
            subjecttotallist = totallist.copy()
            objectfound = candidatefound2
            objectreflist = refdictlist2.copy()
            objectcandlist = canddictlist2.copy()
            objecttotallist = totallist2.copy()

            candidatefound, refdictlist, canddictlist, totallist = getrefdict(newreflist, newcandlist, 'PRED', 'PRED', len(subjecttotallist))
            predicatefound = candidatefound
            predicatereflist = refdictlist.copy()
            predicatecandlist = canddictlist.copy()
            predicatetotallist = totallist.copy()

            switchmatchfound = 'y'
        else:
            switchmatchfound = 'n'

    # Then, let's try to switch subject and predicate
    if ((subjectfound == 'n') and (predicatefound == 'n')) and (switchmatchfound == 'n'):
        refsub = newreference[0]
        candsub = newcandidate[1]

        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)

        reflist = [x.lower() for x in reflist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]
        candlist = [x.lower() for x in candlist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]

        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        # Start with an ngram the full number of words in the candidate
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)

        candidatefound, refdictlist, canddictlist, totallist = getrefdict(newreflist, newcandlist, 'SUB', 'PRED', 0)

        refsub = newreference[1]
        candsub = newcandidate[0]

        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)

        reflist = [x.lower() for x in reflist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]
        candlist = [x.lower() for x in candlist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]

        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        # Start with an ngram the full number of words in the candidate
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)

        candidatefound2, refdictlist2, canddictlist2, totallist2 = getrefdict(newreflist, newcandlist, 'PRED', 'SUB', len(totallist))

        if (candidatefound == 'y') or (candidatefound2 == 'y'):
            subjectfound = candidatefound
            subjectreflist = refdictlist.copy()
            subjectcandlist = canddictlist.copy()
            subjecttotallist = totallist.copy()
            predicatefound = candidatefound2
            predicatereflist = refdictlist2.copy()
            predicatecandlist = canddictlist2.copy()
            predicatetotallist = totallist2.copy()
            switchmatchfound = 'y'
        else:
            switchmatchfound = 'n'

    # Finally, let's try to switch predicate and object
    if ((predicatefound == 'n') and (objectfound == 'n')) and (switchmatchfound == 'n'):
        refsub = newreference[1]
        candsub = newcandidate[2]

        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)

        reflist = [x.lower() for x in reflist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]
        candlist = [x.lower() for x in candlist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]

        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        # Start with an ngram the full number of words in the candidate
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)

        candidatefound, refdictlist, canddictlist, totallist = getrefdict(newreflist, newcandlist, 'PRED', 'OBJ', len(subjecttotallist))

        refsub = newreference[2]
        candsub = newcandidate[1]

        reflist = nltk.word_tokenize(refsub)
        candlist = nltk.word_tokenize(candsub)

        reflist = [x.lower() for x in reflist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]
        candlist = [x.lower() for x in candlist if re.search(r'[' + re.escape(string.punctuation) + r']', x) == None]

        newreflist = reflist.copy()
        newcandlist = candlist.copy()
        # Start with an ngram the full number of words in the candidate
        ngramlength = len(newcandlist)
        newreflist, newcandlist = nonrefwords(newreflist, newcandlist, 1, ngramlength)

        candidatefound2, refdictlist2, canddictlist2, totallist2 = getrefdict(newreflist, newcandlist, 'OBJ', 'PRED', len(subjecttotallist) + len(totallist))

        if (candidatefound == 'y') or (candidatefound2 == 'y'):
            predicatefound = candidatefound
            predicatereflist = refdictlist.copy()
            predicatecandlist = canddictlist.copy()
            predicatetotallist = totallist.copy()
            objectfound = candidatefound2
            objectreflist = refdictlist2.copy()
            objectcandlist = canddictlist2.copy()
            objecttotallist = totallist2.copy()
            switchmatchfound = 'y'
        else:
            switchmatchfound = 'n'


    allrefdict = subjectreflist + predicatereflist + objectreflist
    allcanddict = subjectcandlist + predicatecandlist + objectcandlist
    alltotallist = subjecttotallist + predicatetotallist + objecttotallist

    evaluator = Evaluator([allrefdict], [allcanddict], tags=['SUB', 'PRED', 'OBJ'])

    # Returns overall metrics and metrics for each tag

    results, results_per_tag = evaluator.evaluate()

    return results, results_per_tag

def calculateAllScores(newreflist, newcandlist):
    totalsemevallist = []
    totalsemevallistpertag = []

    for idx, candidate in enumerate(newcandlist):
        if len(newcandlist[idx]) != len(newreflist[idx]):
            differencebetween = abs(len(newcandlist[idx]) - len(newreflist[idx]))
            differencelist = [''] * differencebetween
            if len(newcandlist[idx]) < len(newreflist[idx]):
                newcandlist[idx] = newcandlist[idx] + differencelist
            else:
                newreflist[idx] = newreflist[idx] + differencelist

    for idx, candidate in enumerate(newcandlist):
        candidatesemeval = []
        candidatesemevalpertag = []
        for triple in candidate:
            triplesemeval = []
            triplesemevalpertag = []
            for reference in newreflist[idx]:
                results, results_per_tag = evaluaterefcand(reference, triple)
                triplesemeval.append(results)
                triplesemevalpertag.append(results_per_tag)

            candidatesemeval.append(triplesemeval)
            candidatesemevalpertag.append(triplesemevalpertag)

        totalsemevallist.append(candidatesemeval)
        totalsemevallistpertag.append(candidatesemevalpertag)

    return totalsemevallist, totalsemevallistpertag

def calculateSystemScore(totalsemevallist, totalsemevallistpertag, newreflist, newcandlist):
    selectedsemevallist = []
    selectedsemevallistpertag = []
    alldicts = []

    # Get all the permutations of the number of scores given per candidate, so if there's 4 candidates, but 3 references, this part ensures that one of
    # The four will not be scored
    for idx, candidate in enumerate(newcandlist):
        if len(newcandlist[idx]) > len(newreflist[idx]):
            # Get all permutations
            choosecands = list(itertools.permutations([x[0] for x in enumerate(totalsemevallist[idx])], len(totalsemevallist[idx][0])))
            # The permutations in different orders are not necessary: we only need one order without the number of candidates we're looking at
            choosecands = set([tuple(sorted(i)) for i in choosecands])  # Sort inner list and then use set
            choosecands = list(map(list, choosecands))  # Converting back to list
        else:
            # Otherwise, we're just going to score all candidates
            choosecands = [list(range(len(newcandlist[idx])))]

        # Get all permutations in which the scores can be combined
        if len(newcandlist[idx]) > len(newreflist[idx]):
            choosescore = list(itertools.permutations([x[0] for x in enumerate(totalsemevallist[idx][0])], len(newreflist[idx])))
            choosescore = [list(x) for x in choosescore]
        else:
            choosescore = list(itertools.permutations([x[0] for x in enumerate(totalsemevallist[idx][0])], len(newcandlist[idx])))
            choosescore = [list(x) for x in choosescore]

        # Get all possible combinations between the candidates and the scores
        combilist = list(itertools.product(choosecands, choosescore))

        totaldict = {'totalscore': 0}

        for combination in combilist:
            combiscore = 0
            # Take the combination between the candidate and the score
            zipcombi = list(zip(combination[0], combination[1]))
            collectedsemeval = []
            collectedsemevalpertag = []

            for zc in zipcombi:
                collectedscores = totalsemevallist[idx][zc[0]][zc[1]]
                f1score = statistics.mean([collectedscores['ent_type']['f1'], collectedscores['partial']['f1'], collectedscores['strict']['f1'], collectedscores['exact']['f1']])
                combiscore += f1score

                collectedsemeval.append(collectedscores)
                collectedsemevalpertag.append(totalsemevallistpertag[idx][zc[0]][zc[1]])


            # If the combination is the highest score thus far, or the first score, make it the totaldict
            if (combiscore > totaldict['totalscore']) or (len(totaldict) == 1):
                totaldict = {'totalscore': combiscore, 'combination': combination, 'semevallist': collectedsemeval,
                             'semevalpertaglist': collectedsemevalpertag}

        selectedsemevallist = selectedsemevallist + totaldict['semevallist']
        selectedsemevallistpertag = selectedsemevallistpertag + totaldict['semevalpertaglist']

    print('-----------------------------------------------------------------')
    print('Total scores')
    print('-----------------------------------------------------------------')
    print('Ent_type')
    enttypecorrect = sum([x['ent_type']['correct'] for x in selectedsemevallist])
    enttypeincorrect = sum([x['ent_type']['incorrect'] for x in selectedsemevallist])
    enttypepartial = sum([x['ent_type']['partial'] for x in selectedsemevallist])
    enttypemissed = sum([x['ent_type']['missed'] for x in selectedsemevallist])
    enttypespurious = sum([x['ent_type']['spurious'] for x in selectedsemevallist])
    enttypepossible = sum([x['ent_type']['possible'] for x in selectedsemevallist])
    enttypeactual = sum([x['ent_type']['actual'] for x in selectedsemevallist])
    enttypeprecision = statistics.mean([x['ent_type']['precision'] for x in selectedsemevallist])
    enttyperecall = statistics.mean([x['ent_type']['recall'] for x in selectedsemevallist])
    enttypef1 = statistics.mean([x['ent_type']['f1'] for x in selectedsemevallist])
    print('Correct: ' + str(enttypecorrect) + ' Incorrect: ' + str(enttypeincorrect) + ' Partial: ' + str(enttypepartial) + ' Missed: ' + str(enttypemissed) +
          '\nSpurious: ' + str(enttypespurious) + ' Possible: ' + str(enttypepossible) + ' Actual: ' + str(enttypeactual) + '\nPrecision: ' + str(enttypeprecision) +
          ' Recall: ' + str(enttyperecall) + '\nF1: ' + str(enttypef1))
    print('-----------------------------------------------------------------')
    print('Partial')
    partialcorrect = sum([x['partial']['correct'] for x in selectedsemevallist])
    partialincorrect = sum([x['partial']['incorrect'] for x in selectedsemevallist])
    partialpartial = sum([x['partial']['partial'] for x in selectedsemevallist])
    partialmissed = sum([x['partial']['missed'] for x in selectedsemevallist])
    partialspurious = sum([x['partial']['spurious'] for x in selectedsemevallist])
    partialpossible = sum([x['partial']['possible'] for x in selectedsemevallist])
    partialactual = sum([x['partial']['actual'] for x in selectedsemevallist])
    partialprecision = statistics.mean([x['partial']['precision'] for x in selectedsemevallist])
    partialrecall = statistics.mean([x['partial']['recall'] for x in selectedsemevallist])
    partialf1 = statistics.mean([x['partial']['f1'] for x in selectedsemevallist])
    print('Correct: ' + str(partialcorrect) + ' Incorrect: ' + str(partialincorrect) + ' Partial: ' + str(partialpartial) + ' Missed: ' + str(
        partialmissed) +
          '\nSpurious: ' + str(partialspurious) + ' Possible: ' + str(partialpossible) + ' Actual: ' + str(partialactual) + '\nPrecision: ' + str(
        partialprecision) +
          ' Recall: ' + str(partialrecall) + '\nF1: ' + str(partialf1))
    print('-----------------------------------------------------------------')
    print('Strict')
    strictcorrect = sum([x['strict']['correct'] for x in selectedsemevallist])
    strictincorrect = sum([x['strict']['incorrect'] for x in selectedsemevallist])
    strictpartial = sum([x['strict']['partial'] for x in selectedsemevallist])
    strictmissed = sum([x['strict']['missed'] for x in selectedsemevallist])
    strictspurious = sum([x['strict']['spurious'] for x in selectedsemevallist])
    strictpossible = sum([x['strict']['possible'] for x in selectedsemevallist])
    strictactual = sum([x['strict']['actual'] for x in selectedsemevallist])
    strictprecision = statistics.mean([x['strict']['precision'] for x in selectedsemevallist])
    strictrecall = statistics.mean([x['strict']['recall'] for x in selectedsemevallist])
    strictf1 = statistics.mean([x['strict']['f1'] for x in selectedsemevallist])
    print('Correct: ' + str(strictcorrect) + ' Incorrect: ' + str(strictincorrect) + ' Partial: ' + str(strictpartial) + ' Missed: ' + str(
        strictmissed) +
          '\nSpurious: ' + str(strictspurious) + ' Possible: ' + str(strictpossible) + ' Actual: ' + str(strictactual) + '\nPrecision: ' + str(
        strictprecision) +
          ' Recall: ' + str(strictrecall) + '\nF1: ' + str(strictf1))
    print('-----------------------------------------------------------------')
    print('Exact')
    exactcorrect = sum([x['exact']['correct'] for x in selectedsemevallist])
    exactincorrect = sum([x['exact']['incorrect'] for x in selectedsemevallist])
    exactpartial = sum([x['exact']['partial'] for x in selectedsemevallist])
    exactmissed = sum([x['exact']['missed'] for x in selectedsemevallist])
    exactspurious = sum([x['exact']['spurious'] for x in selectedsemevallist])
    exactpossible = sum([x['exact']['possible'] for x in selectedsemevallist])
    exactactual = sum([x['exact']['actual'] for x in selectedsemevallist])
    exactprecision = statistics.mean([x['exact']['precision'] for x in selectedsemevallist])
    exactrecall = statistics.mean([x['exact']['recall'] for x in selectedsemevallist])
    exactf1 = statistics.mean([x['exact']['f1'] for x in selectedsemevallist])
    print('Correct: ' + str(exactcorrect) + ' Incorrect: ' + str(exactincorrect) + ' Partial: ' + str(exactpartial) + ' Missed: ' + str(
        exactmissed) +
          '\nSpurious: ' + str(exactspurious) + ' Possible: ' + str(exactpossible) + ' Actual: ' + str(exactactual) + '\nPrecision: ' + str(
        exactprecision) +
          ' Recall: ' + str(exactrecall) + '\nF1: ' + str(exactf1))
    print('-----------------------------------------------------------------')
    print('Scores per tag')
    print('-----------------------------------------------------------------')
    print('Subjects')
    print('-----------------------------------------------------------------')
    print('Ent_type')
    subenttypecorrect = sum([x['SUB']['ent_type']['correct'] for x in selectedsemevallistpertag])
    subenttypeincorrect = sum([x['SUB']['ent_type']['incorrect'] for x in selectedsemevallistpertag])
    subenttypepartial = sum([x['SUB']['ent_type']['partial'] for x in selectedsemevallistpertag])
    subenttypemissed = sum([x['SUB']['ent_type']['missed'] for x in selectedsemevallistpertag])
    subenttypespurious = sum([x['SUB']['ent_type']['spurious'] for x in selectedsemevallistpertag])
    subenttypepossible = sum([x['SUB']['ent_type']['possible'] for x in selectedsemevallistpertag])
    subenttypeactual = sum([x['SUB']['ent_type']['actual'] for x in selectedsemevallistpertag])
    subenttypeprecision = statistics.mean([x['SUB']['ent_type']['precision'] for x in selectedsemevallistpertag])
    subenttyperecall = statistics.mean([x['SUB']['ent_type']['recall'] for x in selectedsemevallistpertag])
    subenttypef1 = statistics.mean([x['SUB']['ent_type']['f1'] for x in selectedsemevallistpertag])
    print('Correct: ' + str(subenttypecorrect) + ' Incorrect: ' + str(subenttypeincorrect) + ' Partial: ' + str(subenttypepartial) + ' Missed: ' + str(
        subenttypemissed) +
          '\nSpurious: ' + str(subenttypespurious) + ' Possible: ' + str(subenttypepossible) + ' Actual: ' + str(subenttypeactual) + '\nPrecision: ' + str(
        subenttypeprecision) +
          ' Recall: ' + str(subenttyperecall) + '\nF1: ' + str(subenttypef1))
    print('-----------------------------------------------------------------')
    print('Partial')
    subpartialcorrect = sum([x['SUB']['partial']['correct'] for x in selectedsemevallistpertag])
    subpartialincorrect = sum([x['SUB']['partial']['incorrect'] for x in selectedsemevallistpertag])
    subpartialpartial = sum([x['SUB']['partial']['partial'] for x in selectedsemevallistpertag])
    subpartialmissed = sum([x['SUB']['partial']['missed'] for x in selectedsemevallistpertag])
    subpartialspurious = sum([x['SUB']['partial']['spurious'] for x in selectedsemevallistpertag])
    subpartialpossible = sum([x['SUB']['partial']['possible'] for x in selectedsemevallistpertag])
    subpartialactual = sum([x['SUB']['partial']['actual'] for x in selectedsemevallistpertag])
    subpartialprecision = statistics.mean([x['SUB']['partial']['precision'] for x in selectedsemevallistpertag])
    subpartialrecall = statistics.mean([x['SUB']['partial']['recall'] for x in selectedsemevallistpertag])
    subpartialf1 = statistics.mean([x['SUB']['partial']['f1'] for x in selectedsemevallistpertag])
    print('Correct: ' + str(subpartialcorrect) + ' Incorrect: ' + str(subpartialincorrect) + ' Partial: ' + str(subpartialpartial) + ' Missed: ' + str(
        subpartialmissed) +
          '\nSpurious: ' + str(subpartialspurious) + ' Possible: ' + str(subpartialpossible) + ' Actual: ' + str(subpartialactual) + '\nPrecision: ' + str(
        subpartialprecision) +
          ' Recall: ' + str(subpartialrecall) + '\nF1: ' + str(subpartialf1))
    print('-----------------------------------------------------------------')
    print('Strict')
    substrictcorrect = sum([x['SUB']['strict']['correct'] for x in selectedsemevallistpertag])
    substrictincorrect = sum([x['SUB']['strict']['incorrect'] for x in selectedsemevallistpertag])
    substrictpartial = sum([x['SUB']['strict']['partial'] for x in selectedsemevallistpertag])
    substrictmissed = sum([x['SUB']['strict']['missed'] for x in selectedsemevallistpertag])
    substrictspurious = sum([x['SUB']['strict']['spurious'] for x in selectedsemevallistpertag])
    substrictpossible = sum([x['SUB']['strict']['possible'] for x in selectedsemevallistpertag])
    substrictactual = sum([x['SUB']['strict']['actual'] for x in selectedsemevallistpertag])
    substrictprecision = statistics.mean([x['SUB']['strict']['precision'] for x in selectedsemevallistpertag])
    substrictrecall = statistics.mean([x['SUB']['strict']['recall'] for x in selectedsemevallistpertag])
    substrictf1 = statistics.mean([x['SUB']['strict']['f1'] for x in selectedsemevallistpertag])
    print('Correct: ' + str(substrictcorrect) + ' Incorrect: ' + str(substrictincorrect) + ' Partial: ' + str(substrictpartial) + ' Missed: ' + str(
        substrictmissed) +
          '\nSpurious: ' + str(substrictspurious) + ' Possible: ' + str(substrictpossible) + ' Actual: ' + str(substrictactual) + '\nPrecision: ' + str(
        substrictprecision) +
          ' Recall: ' + str(substrictrecall) + '\nF1: ' + str(substrictf1))
    print('-----------------------------------------------------------------')
    print('Exact')
    subexactcorrect = sum([x['SUB']['exact']['correct'] for x in selectedsemevallistpertag])
    subexactincorrect = sum([x['SUB']['exact']['incorrect'] for x in selectedsemevallistpertag])
    subexactpartial = sum([x['SUB']['exact']['partial'] for x in selectedsemevallistpertag])
    subexactmissed = sum([x['SUB']['exact']['missed'] for x in selectedsemevallistpertag])
    subexactspurious = sum([x['SUB']['exact']['spurious'] for x in selectedsemevallistpertag])
    subexactpossible = sum([x['SUB']['exact']['possible'] for x in selectedsemevallistpertag])
    subexactactual = sum([x['SUB']['exact']['actual'] for x in selectedsemevallistpertag])
    subexactprecision = statistics.mean([x['SUB']['exact']['precision'] for x in selectedsemevallistpertag])
    subexactrecall = statistics.mean([x['SUB']['exact']['recall'] for x in selectedsemevallistpertag])
    subexactf1 = statistics.mean([x['SUB']['exact']['f1'] for x in selectedsemevallistpertag])
    print('Correct: ' + str(subexactcorrect) + ' Incorrect: ' + str(subexactincorrect) + ' Partial: ' + str(subexactpartial) + ' Missed: ' + str(
        subexactmissed) +
          '\nSpurious: ' + str(subexactspurious) + ' Possible: ' + str(subexactpossible) + ' Actual: ' + str(subexactactual) + '\nPrecision: ' + str(
        subexactprecision) +
          ' Recall: ' + str(subexactrecall) + '\nF1: ' + str(subexactf1))
    print('-----------------------------------------------------------------')
    print('Predicates')
    print('-----------------------------------------------------------------')
    print('Ent_type')
    predenttypecorrect = sum([x['PRED']['ent_type']['correct'] for x in selectedsemevallistpertag])
    predenttypeincorrect = sum([x['PRED']['ent_type']['incorrect'] for x in selectedsemevallistpertag])
    predenttypepartial = sum([x['PRED']['ent_type']['partial'] for x in selectedsemevallistpertag])
    predenttypemissed = sum([x['PRED']['ent_type']['missed'] for x in selectedsemevallistpertag])
    predenttypespurious = sum([x['PRED']['ent_type']['spurious'] for x in selectedsemevallistpertag])
    predenttypepossible = sum([x['PRED']['ent_type']['possible'] for x in selectedsemevallistpertag])
    predenttypeactual = sum([x['PRED']['ent_type']['actual'] for x in selectedsemevallistpertag])
    predenttypeprecision = statistics.mean([x['PRED']['ent_type']['precision'] for x in selectedsemevallistpertag])
    predenttyperecall = statistics.mean([x['PRED']['ent_type']['recall'] for x in selectedsemevallistpertag])
    predenttypef1 = statistics.mean([x['PRED']['ent_type']['f1'] for x in selectedsemevallistpertag])
    print('Correct: ' + str(predenttypecorrect) + ' Incorrect: ' + str(predenttypeincorrect) + ' Partial: ' + str(predenttypepartial) + ' Missed: ' + str(
        predenttypemissed) +
          '\nSpurious: ' + str(predenttypespurious) + ' Possible: ' + str(predenttypepossible) + ' Actual: ' + str(predenttypeactual) + '\nPrecision: ' + str(
        predenttypeprecision) +
          ' Recall: ' + str(predenttyperecall) + '\nF1: ' + str(predenttypef1))
    print('-----------------------------------------------------------------')
    print('Partial')
    predpartialcorrect = sum([x['PRED']['partial']['correct'] for x in selectedsemevallistpertag])
    predpartialincorrect = sum([x['PRED']['partial']['incorrect'] for x in selectedsemevallistpertag])
    predpartialpartial = sum([x['PRED']['partial']['partial'] for x in selectedsemevallistpertag])
    predpartialmissed = sum([x['PRED']['partial']['missed'] for x in selectedsemevallistpertag])
    predpartialspurious = sum([x['PRED']['partial']['spurious'] for x in selectedsemevallistpertag])
    predpartialpossible = sum([x['PRED']['partial']['possible'] for x in selectedsemevallistpertag])
    predpartialactual = sum([x['PRED']['partial']['actual'] for x in selectedsemevallistpertag])
    predpartialprecision = statistics.mean([x['PRED']['partial']['precision'] for x in selectedsemevallistpertag])
    predpartialrecall = statistics.mean([x['PRED']['partial']['recall'] for x in selectedsemevallistpertag])
    predpartialf1 = statistics.mean([x['PRED']['partial']['f1'] for x in selectedsemevallistpertag])
    print('Correct: ' + str(predpartialcorrect) + ' Incorrect: ' + str(predpartialincorrect) + ' Partial: ' + str(predpartialpartial) + ' Missed: ' + str(
        predpartialmissed) +
          '\nSpurious: ' + str(predpartialspurious) + ' Possible: ' + str(predpartialpossible) + ' Actual: ' + str(predpartialactual) + '\nPrecision: ' + str(
        predpartialprecision) +
          ' Recall: ' + str(predpartialrecall) + '\nF1: ' + str(predpartialf1))
    print('-----------------------------------------------------------------')
    print('Strict')
    predstrictcorrect = sum([x['PRED']['strict']['correct'] for x in selectedsemevallistpertag])
    predstrictincorrect = sum([x['PRED']['strict']['incorrect'] for x in selectedsemevallistpertag])
    predstrictpartial = sum([x['PRED']['strict']['partial'] for x in selectedsemevallistpertag])
    predstrictmissed = sum([x['PRED']['strict']['missed'] for x in selectedsemevallistpertag])
    predstrictspurious = sum([x['PRED']['strict']['spurious'] for x in selectedsemevallistpertag])
    predstrictpossible = sum([x['PRED']['strict']['possible'] for x in selectedsemevallistpertag])
    predstrictactual = sum([x['PRED']['strict']['actual'] for x in selectedsemevallistpertag])
    predstrictprecision = statistics.mean([x['PRED']['strict']['precision'] for x in selectedsemevallistpertag])
    predstrictrecall = statistics.mean([x['PRED']['strict']['recall'] for x in selectedsemevallistpertag])
    predstrictf1 = statistics.mean([x['PRED']['strict']['f1'] for x in selectedsemevallistpertag])
    print('Correct: ' + str(predstrictcorrect) + ' Incorrect: ' + str(predstrictincorrect) + ' Partial: ' + str(predstrictpartial) + ' Missed: ' + str(
        predstrictmissed) +
          '\nSpurious: ' + str(predstrictspurious) + ' Possible: ' + str(predstrictpossible) + ' Actual: ' + str(predstrictactual) + '\nPrecision: ' + str(
        predstrictprecision) +
          ' Recall: ' + str(predstrictrecall) + '\nF1: ' + str(predstrictf1))
    print('-----------------------------------------------------------------')
    print('Exact')
    predexactcorrect = sum([x['PRED']['exact']['correct'] for x in selectedsemevallistpertag])
    predexactincorrect = sum([x['PRED']['exact']['incorrect'] for x in selectedsemevallistpertag])
    predexactpartial = sum([x['PRED']['exact']['partial'] for x in selectedsemevallistpertag])
    predexactmissed = sum([x['PRED']['exact']['missed'] for x in selectedsemevallistpertag])
    predexactspurious = sum([x['PRED']['exact']['spurious'] for x in selectedsemevallistpertag])
    predexactpossible = sum([x['PRED']['exact']['possible'] for x in selectedsemevallistpertag])
    predexactactual = sum([x['PRED']['exact']['actual'] for x in selectedsemevallistpertag])
    predexactprecision = statistics.mean([x['PRED']['exact']['precision'] for x in selectedsemevallistpertag])
    predexactrecall = statistics.mean([x['PRED']['exact']['recall'] for x in selectedsemevallistpertag])
    predexactf1 = statistics.mean([x['PRED']['exact']['f1'] for x in selectedsemevallistpertag])
    print('Correct: ' + str(predexactcorrect) + ' Incorrect: ' + str(predexactincorrect) + ' Partial: ' + str(predexactpartial) + ' Missed: ' + str(
        predexactmissed) +
          '\nSpurious: ' + str(predexactspurious) + ' Possible: ' + str(predexactpossible) + ' Actual: ' + str(predexactactual) + '\nPrecision: ' + str(
        predexactprecision) +
          ' Recall: ' + str(predexactrecall) + '\nF1: ' + str(predexactf1))
    print('-----------------------------------------------------------------')
    print('Objects')
    print('-----------------------------------------------------------------')
    print('Ent_type')
    objenttypecorrect = sum([x['OBJ']['ent_type']['correct'] for x in selectedsemevallistpertag])
    objenttypeincorrect = sum([x['OBJ']['ent_type']['incorrect'] for x in selectedsemevallistpertag])
    objenttypepartial = sum([x['OBJ']['ent_type']['partial'] for x in selectedsemevallistpertag])
    objenttypemissed = sum([x['OBJ']['ent_type']['missed'] for x in selectedsemevallistpertag])
    objenttypespurious = sum([x['OBJ']['ent_type']['spurious'] for x in selectedsemevallistpertag])
    objenttypepossible = sum([x['OBJ']['ent_type']['possible'] for x in selectedsemevallistpertag])
    objenttypeactual = sum([x['OBJ']['ent_type']['actual'] for x in selectedsemevallistpertag])
    objenttypeprecision = statistics.mean([x['OBJ']['ent_type']['precision'] for x in selectedsemevallistpertag])
    objenttyperecall = statistics.mean([x['OBJ']['ent_type']['recall'] for x in selectedsemevallistpertag])
    objenttypef1 = statistics.mean([x['OBJ']['ent_type']['f1'] for x in selectedsemevallistpertag])
    print('Correct: ' + str(objenttypecorrect) + ' Incorrect: ' + str(objenttypeincorrect) + ' Partial: ' + str(objenttypepartial) + ' Missed: ' + str(
        objenttypemissed) +
          '\nSpurious: ' + str(objenttypespurious) + ' Possible: ' + str(objenttypepossible) + ' Actual: ' + str(objenttypeactual) + '\nPrecision: ' + str(
        objenttypeprecision) +
          ' Recall: ' + str(objenttyperecall) + '\nF1: ' + str(objenttypef1))
    print('-----------------------------------------------------------------')
    print('Partial')
    objpartialcorrect = sum([x['OBJ']['partial']['correct'] for x in selectedsemevallistpertag])
    objpartialincorrect = sum([x['OBJ']['partial']['incorrect'] for x in selectedsemevallistpertag])
    objpartialpartial = sum([x['OBJ']['partial']['partial'] for x in selectedsemevallistpertag])
    objpartialmissed = sum([x['OBJ']['partial']['missed'] for x in selectedsemevallistpertag])
    objpartialspurious = sum([x['OBJ']['partial']['spurious'] for x in selectedsemevallistpertag])
    objpartialpossible = sum([x['OBJ']['partial']['possible'] for x in selectedsemevallistpertag])
    objpartialactual = sum([x['OBJ']['partial']['actual'] for x in selectedsemevallistpertag])
    objpartialprecision = statistics.mean([x['OBJ']['partial']['precision'] for x in selectedsemevallistpertag])
    objpartialrecall = statistics.mean([x['OBJ']['partial']['recall'] for x in selectedsemevallistpertag])
    objpartialf1 = statistics.mean([x['OBJ']['partial']['f1'] for x in selectedsemevallistpertag])
    print('Correct: ' + str(objpartialcorrect) + ' Incorrect: ' + str(objpartialincorrect) + ' Partial: ' + str(objpartialpartial) + ' Missed: ' + str(
        objpartialmissed) +
          '\nSpurious: ' + str(objpartialspurious) + ' Possible: ' + str(objpartialpossible) + ' Actual: ' + str(objpartialactual) + '\nPrecision: ' + str(
        objpartialprecision) +
          ' Recall: ' + str(objpartialrecall) + '\nF1: ' + str(objpartialf1))
    print('-----------------------------------------------------------------')
    print('Strict')
    objstrictcorrect = sum([x['OBJ']['strict']['correct'] for x in selectedsemevallistpertag])
    objstrictincorrect = sum([x['OBJ']['strict']['incorrect'] for x in selectedsemevallistpertag])
    objstrictpartial = sum([x['OBJ']['strict']['partial'] for x in selectedsemevallistpertag])
    objstrictmissed = sum([x['OBJ']['strict']['missed'] for x in selectedsemevallistpertag])
    objstrictspurious = sum([x['OBJ']['strict']['spurious'] for x in selectedsemevallistpertag])
    objstrictpossible = sum([x['OBJ']['strict']['possible'] for x in selectedsemevallistpertag])
    objstrictactual = sum([x['OBJ']['strict']['actual'] for x in selectedsemevallistpertag])
    objstrictprecision = statistics.mean([x['OBJ']['strict']['precision'] for x in selectedsemevallistpertag])
    objstrictrecall = statistics.mean([x['OBJ']['strict']['recall'] for x in selectedsemevallistpertag])
    objstrictf1 = statistics.mean([x['OBJ']['strict']['f1'] for x in selectedsemevallistpertag])
    print('Correct: ' + str(objstrictcorrect) + ' Incorrect: ' + str(objstrictincorrect) + ' Partial: ' + str(objstrictpartial) + ' Missed: ' + str(
        objstrictmissed) +
          '\nSpurious: ' + str(objstrictspurious) + ' Possible: ' + str(objstrictpossible) + ' Actual: ' + str(objstrictactual) + '\nPrecision: ' + str(
        objstrictprecision) +
          ' Recall: ' + str(objstrictrecall) + '\nF1: ' + str(objstrictf1))
    print('-----------------------------------------------------------------')
    print('Exact')
    objexactcorrect = sum([x['OBJ']['exact']['correct'] for x in selectedsemevallistpertag])
    objexactincorrect = sum([x['OBJ']['exact']['incorrect'] for x in selectedsemevallistpertag])
    objexactpartial = sum([x['OBJ']['exact']['partial'] for x in selectedsemevallistpertag])
    objexactmissed = sum([x['OBJ']['exact']['missed'] for x in selectedsemevallistpertag])
    objexactspurious = sum([x['OBJ']['exact']['spurious'] for x in selectedsemevallistpertag])
    objexactpossible = sum([x['OBJ']['exact']['possible'] for x in selectedsemevallistpertag])
    objexactactual = sum([x['OBJ']['exact']['actual'] for x in selectedsemevallistpertag])
    objexactprecision = statistics.mean([x['OBJ']['exact']['precision'] for x in selectedsemevallistpertag])
    objexactrecall = statistics.mean([x['OBJ']['exact']['recall'] for x in selectedsemevallistpertag])
    objexactf1 = statistics.mean([x['OBJ']['exact']['f1'] for x in selectedsemevallistpertag])
    print('Correct: ' + str(objexactcorrect) + ' Incorrect: ' + str(objexactincorrect) + ' Partial: ' + str(objexactpartial) + ' Missed: ' + str(
        objexactmissed) +
          '\nSpurious: ' + str(objexactspurious) + ' Possible: ' + str(objexactpossible) + ' Actual: ' + str(objexactactual) + '\nPrecision: ' + str(
        objexactprecision) +
          ' Recall: ' + str(objexactrecall) + '\nF1: ' + str(objexactf1))
    print('-----------------------------------------------------------------')

def get_classes(newreflist, newcandlist):
    allclasses = newcandlist + newreflist
    allclasses = [item for items in allclasses for item in items]
    allclasses = list(set(allclasses))
    return allclasses

def calculateExactTripleScore(reflist, candlist, classes=None, avg='macro'):
    newreflist = [[string.lower() for string in sublist] for sublist in reflist]
    newcandlist = [[string.lower() for string in sublist] for sublist in candlist]
    #First get all the classes by combining the triples in the candidatelist and referencelist
    if classes is None:
        allclasses = get_classes(newreflist, newcandlist)
    else:
        allclasses = classes
        
    lb = preprocessing.MultiLabelBinarizer(classes=allclasses)
    mcbin = lb.fit_transform(newcandlist)
    mrbin = lb.fit_transform(newreflist)
    print(newcandlist[0])
    print(newreflist[0])
    print(mcbin[0])
    print(mrbin[0])

    precision = precision_score(mrbin, mcbin, average=avg)
    recall = recall_score(mrbin, mcbin, average=avg)
    f1 = f1_score(mrbin, mcbin, average=avg)

    print('Full triple scores')
    print('-----------------------------------------------------------------')
    print('Precision: ' + str(precision) + ' Recall: ' + str(recall) + '\nF1: ' + str(f1))
    return precision, recall, f1

import numpy as np

def main(reffile, candfile):
    reflist, newreflist = getRefs(reffile)
    candlist, newcandlist = getCands(candfile)
    #totalsemevallist, totalsemevallistpertag = calculateAllScores(newreflist, newcandlist)
    #calculateSystemScore(totalsemevallist, totalsemevallistpertag, newreflist, newcandlist)
    classes = get_classes(newreflist, newcandlist)
    print('\n')
    
    print(f"\n#### Macro Averaged ####")
    avg_p, avg_r, avg_f1 = calculateExactTripleScore(reflist, candlist, classes=classes, avg=args.avg)
    n_triples_to_instance = { len(ref): {'refs': [], 'cands': []}
                              for ref in reflist }
    for ref, cand in zip(reflist, candlist):
        n = len(ref)
        n_triples_to_instance[n]['refs'].append(ref)
        n_triples_to_instance[n]['cands'].append(cand)
        
    n_triples_to_performance, n_triples_to_err = {}, {}
    n_triples_to_n_generated_triples = dict(zip(
        n_triples_to_instance.keys(),
        [0 for i in range(len(n_triples_to_instance))]
    ))
    # !!!! PLOT THE DISTRIBUTION OF N GENERATED TRIPLES TO UNDERSTAND WHETHER
    # THE KB ADDITION LEADS TO MORE TRIPLES EXTRACTED !!!!
    for n, refs_cands in n_triples_to_instance.items():
        refs = refs_cands['refs']
        cands = refs_cands['cands']
        print(f"\n#### {n} Triples Sentences ####")
        p, r, f1 = calculateExactTripleScore(refs, cands, classes=classes, avg=args.avg)
        n_triples_to_performance[n] = dict(zip(['p','r','f1'], [p,r,f1]))
        n_triples_to_err[n] = 1 / np.sqrt(len(refs))
        n_triples_to_n_generated_triples[n] = [ len(c) for c in cands ]
    
    n_triples, performance = zip(*sorted(n_triples_to_performance.items(), key=lambda x: x[0]))
    _, errs = zip(*sorted(n_triples_to_err.items(), key=lambda x: x[0]))
    precisions = np.array([ v['p'] for v in performance ])
    recalls = np.array([ v['r'] for v in performance ])
    f1s = np.array([ v['f1'] for v in performance ])

    return n_triples, precisions, recalls, f1s, errs, n_triples_to_n_generated_triples, avg_p, avg_r, avg_f1

    
#main(currentpath + '/Refs.xml', currentpath + '/Cands2.xml')
import argparse, os, re, json
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit

def get_model_info(filename):
    kb = False
    prompt = filename.split('/')[-2].replace('_kb','').replace('prompt_','')
    try:
        model_id = re.search('(?<=generated_triples_)(.*)(?=_temp)', os.path.basename(filename)).group()
        try:
            t = float(re.search('(?<=_temp-)(.*)(?=_kb)', os.path.basename(filename)).group())
            kb = True
        except:
            t = float(re.search('(?<=_temp-)(.*)(?=.xml)', os.path.basename(filename)).group())
    except:
        try:
            model_id = re.search('(?<=generated_triples_)(.*)(?=_kb)', os.path.basename(filename)).group()
            t = 0.1
            kb = True
        except:
            model_id = 'Random'
            kb = True
            t = None
    for pattern in ('huggyllama-', 'tiiuae-', 'chavinlo-'):
        model_id = model_id.replace(pattern, '')
    if model_id != 'Random':
        n_params = model_2_nparams[model_id]
    else:
        n_params = None
    color = model_2_color[model_id]
    if kb:
        model_id += ' (KB'
        if 'complete' in  os.path.basename(filename):
            model_id += ' complete'
        if 'scale' in os.path.basename(filename):
            scale = re.search('(?<=scale-0.)([0-9]+)(?=-)', os.path.basename(filename)).group(0)
            model_id += f' scale=0.{scale}'
        if 'few-shots' in filename:
            model_id += ' few-shots'
        try:
            top_k = re.search('(?<=top-)([0-9]+)(?=\.xml)', os.path.basename(filename)).group(0)
        except:
            top_k = re.search('(?<=scale-)([0-9]+)(?=_)', os.path.basename(filename)).group(0)
        model_id += f' top-{top_k})'
    return model_id, n_params, t, prompt, color

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation + plotting of P,R,F1')
    parser.add_argument('--predictions', nargs='+')
    parser.add_argument('--groundtruth')
    parser.add_argument('--metric', default='F1')
    parser.add_argument('--avg', default='macro')
    parser.add_argument('--colors', nargs='+')
    args = parser.parse_args()

    if args.avg not in ('micro','macro'):
        raise AssertionError('Invalid averaging method, it should be either \'micro\' or \'macro\'')

    if args.metric not in ('P', 'R', 'F1'):
        raise AssertionError('Invalid metric definition, it should be either P, R or F1')

    with open('model_to_n-params.json', 'r') as f:
        model_2_nparams = json.load(f)

    with open('model_to_color.json', 'r') as f:
        model_2_color = json.load(f)

    metrics = {'P': [], 'R': [], 'F1': [], 'ERR': []}
    avg_p, avg_r, avg_f1 = [], [], []
    model_ids, n_params, temperatures, prompts, colors = [], [], [], [], []
    total_n_gen_triples = {}
    for pred in args.predictions:
        print(f'> Evaluating predictions found in: {pred}')
        model_id, n, t, prompt, c = get_model_info(pred)
        model_ids.append(model_id)
        temperatures.append(t)
        n_params.append(n)
        prompts.append(prompt)
        colors.append(c)
        n_triples, p, r, f1, err, n_gen_triples, *avg = main(args.groundtruth, pred)
        avg_p.append(avg[0])
        avg_r.append(avg[1])
        avg_f1.append(avg[2])
        for metric, vals in zip(metrics.keys(), [p,r,f1,err]):
            metrics[metric].append(vals)
        for n, n_gen in n_gen_triples.items():
            if n not in total_n_gen_triples:
                total_n_gen_triples[n] = n_gen
            else:
                total_n_gen_triples[n] += n_gen
    if args.colors is not None:
        colors = [ int(c) for c in args.colors ]
    
    #print(total_n_gen_triples)
    counts, bins = np.histogram(total_n_gen_triples[1], density=True)
    plt.stairs(counts, bins)
    plt.show()

    n_gen_triples = []
    for n_gen in total_n_gen_triples.values():
        n_gen_triples += n_gen
    counts, bins = np.histogram(n_gen_triples, density=True)
    plt.stairs(counts, bins)
    plt.show()
    
    performance_summary = {}
    for model, t, prompt, p, r, f1 in zip(model_ids, temperatures, prompts, avg_p, avg_r, avg_f1):
        performance_summary[f"{model} T={t} prompt={prompt}"] = {'P': p, 'R': r, 'F1': f1}
    with open('performance_summary.json','w') as f:
        json.dump(performance_summary, f, indent=2)

    print('\nModel\t\t\t\t\t\t\tP\tR\tF1')
    print('-------------------------------------------------------------------------------')
    for model, met in performance_summary.items():
        p, r, f1 = (f'{m:.4f}' for m in met.values())
        print(f'{model}\t|\t{p}\t{r}\t{f1}')
    print('-------------------------------------------------------------------------------\n')

    print('# Standard Deviations:')
    stds = []
    for met in (avg_p, avg_r, avg_f1):
        stds.append(np.std(met))
    print('  > P: {:.4f}, R: {:.4f}, F1: {:.4f}'.format(*stds))
        
    # plot performance vs n triples in sentence
    plt.rcParams.update({'font.size': 18})
    plt.figure(figsize=(12,12))
    lines = []
    if len(set(temperatures)) == 1:
        labels = [ f"{mid}" for mid, temp in zip(model_ids, temperatures) ]
        title = f"T = {temperatures[0]}"
    else:
        labels = [ f"{mid} (T={temp})" for mid, temp in zip(model_ids, temperatures) ]
        title = ""

    cmap = matplotlib.colormaps['tab20'] # Alternatively 'Accent'
    width = 0.2 #1 / len(model_ids)
    multiplier = 0
    x_val = np.arange(len(n_triples))
    
    for metric, err, c in zip(metrics[args.metric], metrics['ERR'], colors):
        offset = width * multiplier
        rects = plt.bar(x_val + offset, metric, width, label=labels[multiplier])
        multiplier += 1
        #plt.bar_label(rects, padding=3)
        #lines.append(plt.plot(n_triples, metric, marker='*', markersize=15, linewidth=2, c=cmap(c))[0])
        #upper_lim = metric + err
        #upper_lim[upper_lim > 1] = 1
        #lower_lim = metric - err
        #lower_lim[lower_lim < 0] = 0
        #plt.fill_between(n_triples, lower_lim, upper_lim, alpha=0.1)

    plt.legend()
    #plt.legend(lines, labels)
    #plt.title(title)
    plt.xlabel('N triples in sentence')
    plt.xticks(x_val + width, n_triples)
    plt.ylabel(args.metric)
    plt.savefig('performance_vs_n-triples.pdf', format='pdf', dpi=300)
    plt.show()

    # violin plot kb vs no-kb
    non_kb_perf = []
    kb_perf = [] #{}
    kb_few_shots = [] #{}
    for metric, model in zip(metrics[args.metric], model_ids):
        if 'KB' in model:
            #top_k = re.search('(?<=top-)([0-9]+)(?=\))', os.path.basename(model)).group(0)
            key = 'kb_few_shots' if 'few-shots' in model else 'kb_perf'
            var = vars()[key]
            print(var)
            var.append(metric)
            #if top_k in var.keys():
            #    var[top_k].append(metric)
            #else:
            #    var[top_k] = [metric]
        else:
            non_kb_perf.append(metric)
    kb_perf = np.asarray(kb_perf) #{k: np.asarray(v) for k,v in kb_perf.items()}
    kb_few_shots = np.asarray(kb_few_shots) #{k: np.asarray(v) for k,v in kb_few_shots.items()}
    non_kb_perf = np.asarray(non_kb_perf)

    plt.rcParams.update({'font.size': 44})
    plt.figure(figsize=(10,8))
    if len(non_kb_perf) > 0:
        plt.violinplot(non_kb_perf, positions=n_triples, showmeans=False)
        plt.scatter(n_triples, non_kb_perf.mean(0), label='Zero-Shot')
    #for top_k, data in kb_perf.items():
    if len(kb_perf) > 0:
        plt.violinplot(kb_perf, positions=n_triples, showmeans=False)
        plt.scatter(n_triples, kb_perf.mean(0), label='Zero-Shot + KB')
    #for top_k, data in kb_few_shots.items():
    if len(kb_few_shots) > 0:
        plt.violinplot(kb_few_shots, positions=n_triples, showmeans=False)
        plt.scatter(n_triples, kb_few_shots.mean(0), label='Few-Shots')
    plt.xlabel('N triplets in sentence')
    plt.ylabel(args.metric)
    #plt.legend()
    plt.tight_layout()
    plt.savefig('violin_plot.pdf', format='pdf', dpi=300)
    plt.show()

    
            
    # plot performance vs n parameters
    nparams_to_perf = np.asarray(sorted(zip(n_params, avg_p, avg_r, avg_f1), key=lambda x: x[0]))
    plt.rcParams.update({'font.size': 24})
    plt.figure(figsize=(12,12))
    #plt.scatter(nparams_to_perf[:,0], nparams_to_perf[:,1], marker='*', c='blue', s=150)
    #plt.scatter(nparams_to_perf[:,0], nparams_to_perf[:,2], marker='*', c='orange', s=150)
    plt.scatter(nparams_to_perf[:,0], nparams_to_perf[:,3], marker='*', c='green', s=150)
    #for i, curve in zip(range(1,4), [('Precision', 'blue'), ('Recall', 'orange'), ('F1', 'green')]):
        #coeff = np.polyfit(nparams_to_perf[:,0], nparams_to_perf[:,i], deg=1)
        #f = lambda x: coeff[0]*x + coeff[1]
        #plt.plot(np.log(nparams_to_perf[:,0]), f(nparams_to_perf[:,0]), label=curve[0], c=curve[1], linewidth=2)
    #plt.plot(nparams_to_perf[:,0], f(nparams_to_perf[:,0]), label='Recall')
    #plt.plot(nparams_to_perf[:,0], f(nparams_to_perf[:,0]), label='F1')
    plt.xscale('log')
    plt.xlabel('N Parameters')
    plt.ylabel('Performance')
    plt.legend()
    plt.savefig('performance_vs_n-params.pdf', format='pdf', dpi=300)
    plt.show()

    # plot performance vs temperature
    temp_to_perf = np.asarray(sorted(zip(temperatures, avg_p, avg_r, avg_f1), key=lambda x: x[0]))
    plt.rcParams.update({'font.size': 24})
    plt.figure(figsize=(12,12))
    plt.scatter(temp_to_perf[:,0], temp_to_perf[:,1], marker='*', s=150)
    plt.scatter(temp_to_perf[:,0], temp_to_perf[:,2], marker='*', s=150)
    plt.scatter(temp_to_perf[:,0], temp_to_perf[:,3], marker='*', s=150)
    plt.plot(temp_to_perf[:,0], temp_to_perf[:,1], label='Precision', linewidth=2)
    plt.plot(temp_to_perf[:,0], temp_to_perf[:,2], label='Recall', linewidth=2)
    plt.plot(temp_to_perf[:,0], temp_to_perf[:,3], label='F1', linewidth=2)
    plt.xlabel('Temperature')
    plt.ylabel('Performance')
    plt.legend()
    plt.savefig('performance_vs_temperature.pdf', format='pdf', dpi=300)
    plt.show()
