import numpy as np;
import random as rd;
from os.path import join;
from numpy.linalg import norm;
import sys;
import statistics;
from operator import add;

class KMeansClustering:
	
	def __init__(self, K, maxIterations, epsilon, numRuns):
		self.K = K;
		self.maxIterations = maxIterations;
		self.epsilon = epsilon;
		self.numRuns = numRuns;
		self.data = [];
		self.dataCentroids = [];
		self.centroids = [];
		self.loadData();
		
		self.sseAvg = 0;
		for run in range(numRuns):
			self.sseAvg += self.fit();
			self.dataCentroids = [];
			self.centroids = [];
		self.sseAvg /= numRuns;
		print("Average sse val for "+str(self.K)+": ");
		print(self.sseAvg)
		

	def loadData(self):
		dataDirectory = "Data";
		dataFile = "seeds.txt";
		with open(join(dataDirectory, dataFile), 'r') as file:
			for line in file:
				self.data.append([float(x) for x in line.strip().split()]);
		self.dataPoints = [None]*len(self.data);
					
	def initializeCentroids(self):
		centroids = rd.sample(range(len(self.data)), self.K);
		#print(centroids)
		for x in centroids:
			self.dataCentroids.append(self.data[x]);
			self.centroids.append([]);
		#print(self.centroids);
		
	def updateDistances(self):
		for x in range(len(self.centroids)):
			self.centroids[x] = [];
		#print(self.centroids)
		for x in range(len(self.data)):
			minDistance = sys.maxsize;
			minCentroid = 0;
			self.dataPoints[x] = [];
			for centroid in range(len(self.centroids)):
				a = np.array(self.data[x]);
				b = np.array(self.dataCentroids[centroid])
				distance = norm(a-b);
				self.dataPoints[x].append((centroid, distance));
				if distance < minDistance:
					minDistance = distance;
					minCentroid = centroid;
			#print(x)
			#print(minDistance)
			#print(minCentroid)
			self.centroids[minCentroid].append(x);
		#print(self.centroids)
			
		#print(self.dataPoints[180]);
		#print(self.data[180]);
		#print(self.data[self.dataPoints[180][0][0]])
		#print(self.data[self.dataPoints[180][1][0]])
		#print(self.data[self.dataPoints[180][2][0]])
		#for x in range(210):
			#print(str(x)+": "+str(self.dataPoints[x]))
	
	def updateCentroids(self):
		#print("HIII")
		#print(self.centroids);
		for oldCentroidIdx in range(self.K):
			sumPoints = [0]*len(self.data[0]);
			for x in self.centroids[oldCentroidIdx]:
				sumPoints = list(map(add, sumPoints, self.data[x]));
			length = len(self.centroids[oldCentroidIdx]);
			newCentroid = [elem/length for elem in sumPoints];

			#print(newCentroid);
			self.dataCentroids[oldCentroidIdx] = newCentroid;
		
	def computeSSE(self):
		sumDistances = 0;
		for centroid in range(self.K):
			for point in self.centroids[centroid]:
				a = np.array(self.data[point]);
				b = np.array(self.dataCentroids[centroid]);
				distance = norm(a-b);
				sumDistances += distance**2;
		return sumDistances;

	def fit(self):
		self.initializeCentroids();
		sseVals = [];
		for x in range(self.maxIterations):	
			self.updateDistances();
			self.updateCentroids();
			sse = self.computeSSE();
			sseVals.append(sse);
			if len(sseVals) > 1 and abs(sseVals[len(sseVals)-1] - sseVals[len(sseVals)-2]) < self.epsilon:
				#print(sseVals[len(sseVals)-1]);
				#print(sseVals[len(sseVals)-2])
				#print("ended by: "+str(x))
				break;
		#print(len(self.centroids))
		#print(self.centroids)
		#print(self.dataCentroids)
		#print(sseVals)
		return sseVals[len(sseVals)-1];

def main():
	K = [3, 5, 7];
	maxIterations = 100;
	epsilon = 0.001;
	numRuns = 10;
	for kval in K:	
		KMeans = KMeansClustering(kval, maxIterations, epsilon, numRuns);
	
if __name__ == '__main__':
	main();
