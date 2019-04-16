from collections import defaultdict


class vertex:
    vertex_index = defaultdict(list)
    def __init__(self,vert,neighbors,degree,color,sameDegreeVertices):
        self.vertexNo = vert
        self.neighborList = neighbors
        self.degree = degree
        self.color = color
        self.verticesOfSameDegree = sameDegreeVertices
        vertex.vertex_index[vert].append(self)
    def printVertex(self):
        print("vertexNo",self.vertexNo)
        print("degree",self.degree)
        print("neighbourVertices",self.neighbourVertices)

    @classmethod
    def findByVertexNo(cls, vert):
        return vertex.vertex_index[vert]