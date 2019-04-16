class Node:
    def __init__(self,data):
        self.data = data
        self.next = None
        self.prev = None


class doublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None


    def getNode(self,index):
        temp = self.head
        for iterator in range(index):
            if temp is None:
                return None
            temp = temp.next
        return temp

    def insertAfter(self,givenNode,newNode):
        newNode.prev = givenNode
        if givenNode.next is None:
            self.tail = newNode
        else:
            newNode.next = givenNode.next
            newNode.next.prev = newNode
        givenNode.next = newNode

    def insertBefore(self, givenNode, new_node):
        new_node.next = givenNode
        if givenNode.prev is None:
            self.head = new_node
        else:
            new_node.prev = givenNode.prev
            new_node.prev.next = new_node
        givenNode.prev = new_node

    def deleteNode(self, node):
        if node.prev is None:
            self.head = node.next
        else:
            node.prev.next = node.next

        if node.next is None:
            self.tail = node.prev
        else:
            node.next.prev = node.prev

