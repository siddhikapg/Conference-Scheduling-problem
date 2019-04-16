import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import math
from mergesort import mergeSort
from collections import defaultdict
import networkx as nx
from vertex import vertex
import time

colors_of_nodes = {}

##----------------------------------------------------------------------------------------------------------------------
#Description                        Calculates session conflicts for each attendee
#Input : attendeeSessionsList       Master list of each attendee and his sessions
#                                   -Example [[1,2,3],[3,4,5],[1,3,2],[3,4,1]]
#                                   in this list index indicates the attendee and content at index is the
#                                   list of his sessions.
#        numOfAttendees             Number of attendees
#        sessionsPerAttendee        Number of sessions an attendee will attend
#Output : conflictsList             This list has session conflicts for all attendeed
##----------------------------------------------------------------------------------------------------------------------
def computeConflicts(attendeeSessionsList, numOfAttendees):
    conflictsList = []
    for i in range(numOfAttendees):
         conflictsList.extend(list(itertools.combinations(attendeeSessionsList[i],2)))
        #append each conflict tuple from tempList to the master list conflictsList

    return conflictsList



##----------------------------------------------------------------------------------------------------------------------
# Description                           Removes duplicates from the conflicts list
# Input :   conflictsList               This is the master list of all the conflicts for all attendees
# Output :  uniqueConflictsList         This is a list that contains unique session conflicts
##----------------------------------------------------------------------------------------------------------------------
def removeDuplicates(conflictsList):
    #Here we sort every tuple in the list, so that conflicts like (2,4) and (4,2) are identified and
    # one of them is removed as a duplicate
    #conflictsList = sorted(sorted(conflictsList[i]) for i in range(len(conflictsList)))
    #conflictsList = sorted((conflictsList[i]) for i in range(len(conflictsList)))
    print("conflictsList :",conflictsList)
    uniqueConflictsList = set(conflictsList[i] for i in range(len(conflictsList)))
    print("uniqueConflictsList",uniqueConflictsList)
    return list(uniqueConflictsList)

##----------------------------------------------------------------------------------------------------------------------
# Description                           Removes duplicates from the conflicts list
# Input :   conflictsList               This is the master list of all the conflicts for all attendees
# Output :  uniqueConflictsList         This is a list that contains unique session conflicts
##----------------------------------------------------------------------------------------------------------------------
def removeDuplicatesNSQ(conflictsList):

    uniqueList = []
    for i in range(len(conflictsList)):
        if(len(uniqueList) == 0):
            uniqueList.append(conflictsList[i])
        else:
            for j in range(len(uniqueList)):
                if(uniqueList[j] == conflictsList[i]):
                    duplicate = 1
                    break
            if duplicate == 0:
                uniqueList.append(conflictsList[i])
    return uniqueList



##----------------------------------------------------------------------------------------------------------------------
#Description                        This method generates sessions per attendee as uniform random numbers
#Input : numOfSessions              Total number of sessions
#        numOfAttendees             Total number of attendees
#        sessionsPerattendee        Sessions per each attendee
#Output :sessionList                It returns a masterlist which has index as the attendee and the content of the list at
#                                   index is the sessions he will attend
##------------------------------------------------------------------------------------------------------------------------
def uniformRandom(numOfSessions, numOfAttendees,sessionsPerAttendee,listForFrequency):
    sessionList = []
    #This list will contain all the session occurrences for all attendees
    #This will be used to count the frequency of sessions to plot a histogram

    for j in range(numOfAttendees):
        sessions = random.sample(range(1,numOfSessions),sessionsPerAttendee)
        #sort the sessions list
        mergeSort(sessions,len(sessions))

        sessionList.append(sessions)
        listForFrequency.extend(sessions)
    return sessionList

#-----------------------------------------------------------------------------------------------------------------------
#Description                        This method generates sessions per attendee as a skewed distributed random number
#Input : N                          Total number of sessions
#        alpha                      Total number of attendees
#        loc                        Sessions per each attendee
#        scale                      scale
#Output :sessionList                It returns a list of skewed random sessions
#                                   index is the sessions he will attend
##----------------------------------------------------------------------------------------------------------------------
def computeSkew(N, alpha=0, loc=1, scale=100):
    sigma = alpha / np.sqrt(1.0 + alpha**2)
    u0 = np.random.randn(N)
    v = np.random.randn(N)
    u1 = (sigma*u0 + np.sqrt(1.0 - sigma**2)*v) * scale
    u1[u0 < 0] *= -1
    u1 = u1 + loc
    return u1

##----------------------------------------------------------------------------------------------------------------------
#Description                        This method generates sessions per attendee as skewed random numbers
#Input : numOfSessions              Total number of sessions
#        numOfAttendees             Total number of attendees
#        sessionsPerattendee        Sessions per each attendee
#        listForFrequency           This acts as an output parameter which has consolidated list of sessions
#Output :sessionList                It returns a masterlist which has index as the attendee and the content of the list at
#                                   index is the sessions he will attend
##------------------------------------------------------------------------------------------------------------------------
def skewedRandom(numOfSessions, numOfAttendees,sessionsPerAttendee,listForFrequency):
    sessionList = []

    for j in range(numOfAttendees):
        sessions=[]
        for j in range(sessionsPerAttendee):
            sessions.append(math.floor(abs(random.random() - random.random()) * (numOfSessions-1) + 1))

        mergeSort(sessions,len(sessions))
        sessionList.append(sessions)
        listForFrequency.extend(sessions)
    return sessionList

##------------------------------------------------------------------------------------------------------------------------
#Description                        This method generates sessions per attendee as skewed random numbers such that
#                                   the attendees are biased towards sessions in the middle region of sessions
#Input : numOfSessions              Total number of sessions
#        numOfAttendees             Total number of attendees
#        sessionsPerattendee        Sessions per each attendee
#        listForFrequency           This acts as an output parameter which has consolidated list of sessions
#Output :sessionList                It returns a masterlist which has index as the attendee and the content of the list at
#                                   index is the sessions he will attend
##------------------------------------------------------------------------------------------------------------------------
def skewedRandomOpp(numOfSessions, numOfAttendees,sessionsPerAttendee,listForFrequency):
    sessionList = []

    for j in range(numOfAttendees):
        sessions=[]
        for j in range(sessionsPerAttendee):
            sessions.append(math.floor(abs((random.random() + random.random())/2) *(numOfSessions) + 1))

        mergeSort(sessions,len(sessions))
        sessionList.append(sessions)
        listForFrequency.extend(sessions)
    return sessionList

#-------------------------------------------------------------------------------------------------------------------------
#Description                        This method generates sessions per attendee as two-tiered random numbers
#Input : numOfSessions              Total number of sessions
#        numOfAttendees             Total number of attendees
#        sessionsPerattendee        Sessions per each attendee
#        listForFrequency           This acts as an output parameter which has consolidated list of sessions
#Output :sessionList                It returns a masterlist which has index as the attendee and the content of the list at
#                                   index is the sessions he will attend
#-------------------------------------------------------------------------------------------------------------------------
def twoTieredRandom(numOfSessions, numOfAttendees,sessionsPerAttendee,listForFrequency):
    sessionList = []
    #in this distribution, 50% of the session an attendee will attend
    #will be from the 10% of the sessions and that too lower numbered.
    #rest of the 50% of sessions will be from the other 90% of the sessions
    if(numOfAttendees % 2 ==0):
        attendeeIndex = int(round(numOfAttendees/2))
    else:
        attendeeIndex = int(round((numOfAttendees+1)/4))

    if (numOfSessions < 10):
        sessionIndex = 2
    else:
        if(numOfSessions % 10 ==0):
            sessionIndex = int(numOfSessions /10)+1
        else:
            sessionIndex = int(numOfSessions % 10)

    if(sessionsPerAttendee % 2 == 0):
        halfSessions = int(sessionsPerAttendee /2)
    else:
        halfSessions = int((sessionsPerAttendee+1)/2)

    for i in range(numOfAttendees):
        # get the 50% of random sessions from first 10% of the sessions
        # here, sessionindex is at 10% of the total sessions

        if halfSessions>sessionIndex :
            sessions = random.sample(range(1,halfSessions+1),halfSessions)
        else:
            sessions = random.sample(range(1, sessionIndex + 1), halfSessions)

        #now, get rest of the 50% of random sessions from 90% of sessions
        sessions.extend(random.sample(range(sessionIndex+1,numOfSessions), sessionsPerAttendee-halfSessions))

        mergeSort(sessions, len(sessions))
        sessionList.append(sessions)
        listForFrequency.extend(sessions)
    return sessionList

#----------------------------------------------------------------------------------------------------------------------------
#Description                        This method generates adjacency list based on the session conflicts
#Input : sessionConflictsList       This is a list of session conflicts for all the attendees
#Output: graph                      This is the graph structure containing adjacency list of session conflicts for all attendees
#----------------------------------------------------------------------------------------------------------------------------
def calculateAdjacencyList(List1):
    graph = defaultdict(list)

    #traverse the consolidated list of unique session conflicts and
    #populate the adjacency list
    for i in range(len(List1)):
        graph[List1[i][0]].append(List1[i][1])
        graph[List1[i][1]].append(List1[i][0])

    #print("graph",graph)
    return graph
#-----------------------------------------------------------------------------------------------------------------------
#Description:               This method creates the output arrays P and E
#Input: adjacencyGraph      The graph containing adjacency list of session conflicts
#       eArray1             This acts as an output variable for E array
#       pArray1             This acts as an output variable for P array
#-----------------------------------------------------------------------------------------------------------------------
def createOutputArrays(adjacencyGraph,eArray1,pArray1):
    eArrayIndex = 0
    for i,k in adjacencyGraph.items():
        temp1 = k
        pArray1.append(eArrayIndex)
        for j in range(len(temp1)):
            eArray1.insert(eArrayIndex,temp1[j])
            eArrayIndex +=1

#-----------------------------------------------------------------------------------------------------------------------
#Description:               This method prints the output arrays P and E to a text file
#Input: eArrayOutput        This acts as an output variable for E array
#       pArrayOutput        This acts as an output variable for P array
#-----------------------------------------------------------------------------------------------------------------------
def printToFile(eArrayOutput,pArrayOutput):
    f1 = open("/Users/Ketan/Documents/Siddhika/SMU/Courses/Fall 2018/7350 Algorithm Engineering/Project/output.txt","w+")
    f1.write("P[]\n")
    for j in pArrayOutput:
        f1.write("%s\n" % j)

    f1.write("\n------------------\n")

    f1.write("E[]\n")
    for i in eArrayOutput:
        f1.write("%s\n" % i)

    f1.close()


#-----------------------------------------------------------------------------------------------------------------------
#Description:               This method creates graph that has degrees as key along with list of vertices
#                           of each degree
#Input: adjacencyGraph      This is the graph that contains data of all the sessions conflicts.
#output : degreeList        This is another defaultdict that has list of vertices with different degrees
#-----------------------------------------------------------------------------------------------------------------------
def createDegreeList(adjacencyGraph):
    degreeGraph = defaultdict(list)
    count = 0
    #print("graph",adjacencyGraph)
    for i, k in adjacencyGraph.items():
    #length of list k will give the degree of vertex
        if count < len(k):
            while len(k)!= count:
                degreeGraph[count] = []
                count += 1
        degreeGraph[len(k)].append(i)
        if len(k) >= count:
            count = len(k)+ 1
    #sort the degree graph in increasing order
    finalDegreeGraph = sorted(degreeGraph.items(),key = lambda k: k[0])
    return finalDegreeGraph


#-----------------------------------------------------------------------------------------------------------------------
#Description:               This method creates a list that enlists vertex and it's neighbour vertices
#Input: adjacencyGraph      This is the graph that contains data of all the sessions conflicts.
#output : degreeList        This is list that enlists vertex and it's neighbour vertices
#-----------------------------------------------------------------------------------------------------------------------
def createVertexList(adjacencyGraph, degreeList):
    vertexList = []
    for i, k in adjacencyGraph.items():
        temp = vertex(i,k,len(k),0,degreeList[len(k)][1])
        vertexList.append(temp)
    return vertexList


#-----------------------------------------------------------------------------------------------------------------------
#Description:               This method colors the graph vertices in the order of vertex such that
#                           the vertex with smallest degree is colored last
#Input: adjacencyGraph      This is the graph that contains data of all the sessions conflicts.
#-----------------------------------------------------------------------------------------------------------------------
def smallestLast(adjacencyGraph):
    outputList = []

    #create the degree list
    degreeList = createDegreeList(adjacencyGraph)

    #create the vertex list
    vertexList = createVertexList(adjacencyGraph,degreeList)

    outputGraph = defaultdict(list)

    iterator = 0

    degreeWhenDeletedList = defaultdict(list)

    #get the maximum degree in the degree list.
    # We'll need it later.
    maxDegree = max(degreeList, key=lambda x:x[0])[0]

    while len(outputList) is not len(vertexList):
        backUp = 0
        backupValue = 0
        minDegreeItem = degreeList[iterator]

        #Here we have all the vertices of this degree
        VerticesOfThisDegree = minDegreeItem[1]

        if len(VerticesOfThisDegree) > 0:

            for member in VerticesOfThisDegree:
                #get the vertex structure of this vertex
                vertexStruct = vertex.findByVertexNo(member)

                if vertexStruct[0].color == 0:

                    #Mark color of the vertex 1 to indicate it has been deleted
                    vertexStruct[0].color = 1
                    outputList.append(member)
                    degreeWhenDeletedList[vertexStruct[0].vertexNo] = vertexStruct[0].degree
                    # send it's neighbours 1 degree up
                    neighbors = vertexStruct[0].neighborList

                    #add the vertex to output graph
                    outputGraph[member] = neighbors

                    #Let's update it's neighbors and their location in degreeList
                    for innerIter in range(len(neighbors)):
                        neighborVertex = vertex.findByVertexNo(neighbors[innerIter])

                        # check if this neighbor is deleted/processed already
                        if neighborVertex[0].color == 0:
                            #Move it to the location of it's current degree -1
                            degreeList[neighborVertex[0].degree][1].remove(neighborVertex[0].vertexNo)
                            degreeList[neighborVertex[0].degree - 1][1].append(neighborVertex[0].vertexNo)
                            neighborVertex[0].degree = neighborVertex[0].degree - 1

                            #If any neighbor is moved to degree smaller than current iterator,
                            # we need to know where to BACK UP !
                            if neighborVertex[0].degree < iterator and backUp == 0:
                                backUp = 1
                                backupValue = neighborVertex[0].degree

        # It's time to decide which degree to process next
        # SO.... first check if any vertex had backed up to degree lesser than current iterator
        if backUp == 1:
            iterator = backupValue
        else:
            #You might think why would we EVER need this check ... trust me, WE DO!!
            #And we need to go back to the start of the list in this case
            if iterator >= maxDegree:
                iterator = 0
            else:
                iterator = iterator + 1

    # Now reverse the graph...... BECAUSE.... It's smallest vertex LAST !!!!
    reversedGraph = defaultdict(list)

    while len(outputGraph) > 0:
        tempItem = outputGraph.popitem()
        reversedGraph[tempItem[0]] = tempItem[1]

    index = len(degreeWhenDeletedList)+1
    for x,y in degreeWhenDeletedList.items():
        plt.plot(index,y,'ro')
        plt.text(index,y,x)
        index = index - 1
    plt.xlabel("order of coloring")
    plt.ylabel("Degree when deleted")
    plt.show()

    return reversedGraph


#-----------------------------------------------------------------------------------------------------------------------
#Description:               This method pics the graph vertices in uniform random fashion
#Input: adjacencyGraph      This is the graph that contains data of all the sessions conflicts.
#-----------------------------------------------------------------------------------------------------------------------
def pickRandomVertex(adjacencyGraph):
    vertexListTemp = []
    outGraph = defaultdict(list)
    for i, k in adjacencyGraph.items():
        vertexListTemp.append(i)
    outputListTemp = []
    while len(vertexListTemp) > 0:
        choice = random.choice(vertexListTemp)
        neighbors = adjacencyGraph.get(choice)
        outGraph[choice] = neighbors
        outputListTemp.append(choice)
        vertexListTemp.remove(choice)
    return outGraph


#-----------------------------------------------------------------------------------------------------------------------
#Description:               This method colors the graph vertices in the order of vertex such that
#                           the vertex with largest degree is colored first
#Input: adjacencyGraph      This is the graph that contains data of all the sessions conflicts.
#-----------------------------------------------------------------------------------------------------------------------
def largestVertexFirst(adjacencyGraph):

    outputList = []

    #create the degree list
    degreeListLarge = createDegreeList(adjacencyGraph)

    #create the vertex list
    vertexListLarge = createVertexList(adjacencyGraph,degreeListLarge)

    outputGraph = defaultdict(list)

    degreeWhenDeletedList = defaultdict(list)

    #get the maximum degree in the degree list.
    maxDegree = max(degreeListLarge, key=lambda x:x[0])[0]
    iterator = maxDegree
    while len(outputList) is not len(vertexListLarge):
        maxDegreeItem = degreeListLarge[iterator]

        #Here we have all the vertices of this degree
        VerticesOfThisDegree = maxDegreeItem[1]

        if len(VerticesOfThisDegree) > 0:

            for member in VerticesOfThisDegree:
                #get the vertex structure of this vertex
                vertexStructLarge = vertex.findByVertexNo(member)
                #print("next vertex", member)
                if vertexStructLarge[0].color == 0:

                    #Mark color of the vertex 1 to indicate it has been deleted
                    vertexStructLarge[0].color = 1
                    outputList.append(member)
                    degreeWhenDeletedList[vertexStructLarge[0].vertexNo] = vertexStructLarge[0].degree
                    # send it's neighbours 1 degree up
                    neighbors = vertexStructLarge[0].neighborList

                    #add the vertex to output graph
                    outputGraph[member] = neighbors

                    #Let's update it's neighbors and their location in degreeList
                    for innerIter in range(len(neighbors)):
                        neighborVertex = vertex.findByVertexNo(neighbors[innerIter])

                        # check if this neighbor is deleted/processed already
                        if neighborVertex[0].color == 0:
                            #Move it to the location of it's current degree -1
                            degreeListLarge[neighborVertex[0].degree][1].remove(neighborVertex[0].vertexNo)
                            degreeListLarge[neighborVertex[0].degree - 1][1].append(neighborVertex[0].vertexNo)
                            neighborVertex[0].degree = neighborVertex[0].degree - 1

        iterator = iterator - 1

    #print("output list :", outputList)
    #print("output graph before : ", outputGraph)

    index = len(degreeWhenDeletedList)+1
    for x,y in degreeWhenDeletedList.items():
        plt.plot(index,y,'ro')
        plt.text(index,y,x)
        index = index - 1
    plt.xlabel("order of coloring")
    plt.ylabel("Degree when deleted")
    plt.show()

    return outputGraph

#-----------------------------------------------------------------------------------------------------------------------
#Description:               This method checks if the color of a graph node conflicts with its neighbors
#Input: adjacencyGraph      This is the graph that contains data of all the sessions conflicts.
#-----------------------------------------------------------------------------------------------------------------------
def checkNeighborsColor(node, color,graph):
   for neighbor in graph.neighbors(node):
       color_of_neighbor = colors_of_nodes.get(neighbor,None)
       if color_of_neighbor == color:
          return False

   return True


#-----------------------------------------------------------------------------------------------------------------------
#Description:               This method pics the graph vertices in the order of vertex such that
#                           the vertex with smallest degree is colored last
#Input: adjacencyGraph      This is the graph that contains data of all the sessions conflicts.
#-----------------------------------------------------------------------------------------------------------------------
def colorForEachNode(node,graph):
    colors = ['Red', 'Blue', 'Green', 'Yellow', 'Black', 'Pink', 'Orange', 'White', 'Gray', 'Purple', 'Brown', 'Navy',
              'Cyan', 'Beige',
              'Azure']

    for color in colors:
        if checkNeighborsColor(node,color,graph):
            return color

#-----------------------------------------------------------------------------------------------------------------------
#Description:               This method colors the graph vertices
#Input: adjacencyGraph      This is the graph that contains data of all the sessions conflicts.
#-----------------------------------------------------------------------------------------------------------------------
def colorTheGraph(adjGraph):

    colorArray = []
    tempGraph = nx.Graph(adjGraph)
    for node in tempGraph.nodes():
        # for node in nodeList:
        colors_of_nodes[node] = colorForEachNode(node,tempGraph)

    for iter1, iter2 in colors_of_nodes.items():
        colorArray.append(iter2)

    pos = nx.shell_layout(tempGraph)
    nx.draw_networkx(tempGraph, pos=pos, node_color=colorArray)

    plt.show()



#-----------------------------------TEST CODE---------------------------------------------#

print("\nPlease provide following input :")
numOfSessions = input("The number of sessions (N):")
numOfSessions = int(numOfSessions)
numOfAttendees = input("The number of attendees (S):")
numOfAttendees = int(numOfAttendees)
sessionsPerAttendee = input("The number os sessions per attendee (K):")
sessionsPerAttendee = int(sessionsPerAttendee)
distribution = input("The distribution of sessions : ")
listForFrequency = []
if (distribution == "UNIFORM"):
    sessionList = uniformRandom(numOfSessions, numOfAttendees, sessionsPerAttendee,listForFrequency)
if (distribution == "TIERED"):
    sessionList = twoTieredRandom(numOfSessions, numOfAttendees, sessionsPerAttendee,listForFrequency)
if (distribution == "SKEWED"):
    sessionList = skewedRandom(numOfSessions, numOfAttendees, sessionsPerAttendee,listForFrequency)
if (distribution == "CUSTOM"):
    sessionList = skewedRandomOpp(numOfSessions, numOfAttendees, sessionsPerAttendee,listForFrequency)

# Process the session list to get session conflicts

list1 = computeConflicts(sessionList,numOfAttendees)
uniqueList = removeDuplicates(list1)
adjacencyList = calculateAdjacencyList(uniqueList)


eArray =[]
pArray = []
createOutputArrays(adjacencyList,eArray,pArray)

printToFile(eArray,pArray)

#---------------------------------OUTPUT---------------------------------------------
print("\n-----------------------------OUTPUT-------------------------------------")
print("\nNumber of sessions (N): ", numOfSessions)
print("\nNumber of Attendees (S): ",numOfAttendees)
print("\nNumber of sessions per attendee (K):",sessionsPerAttendee)
print("\nTotal number of pair-wise session conflicts (T):",len(list1))
print("\nNumber of distinct pair-wise session conflicts (M):",len(uniqueList))
print("\nDistribution : ",distribution)
print("-------------------------------------------------------------------------")

plt.hist(listForFrequency)
plt.xlabel('Sessions', fontsize=12)
plt.ylabel('No. of Attendees', fontsize=12)
plt.show()


#-------------------------------------graph draw test code --------------------------------------
randomList = pickRandomVertex(adjacencyList)

sortedGraph = smallestLast(adjacencyList)
print("\nadjacency list  : ",adjacencyList)

randomGraph = pickRandomVertex(adjacencyList)

largestFirstGraph = largestVertexFirst(adjacencyList)
print("\nlargest first : ",largestFirstGraph)

colorTheGraph(largestFirstGraph)