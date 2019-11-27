import numpy as np;
import random as rd;
from os.path import join;
from numpy.linalg import norm;
import sys;
import statistics;
from operator import add;
from sklearn.cluster import KMeans

## The class implements K-means clustering algorithm for given K
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
			if run < numRuns-1:
				self.dataCentroids = [];
				self.centroids = [];
		self.sseAvg /= numRuns;
		print("Average SSE value for "+str(self.K)+": " + str(self.sseAvg));
		

	## Load the given data (seeds.txt) in Data folder
	def loadData(self):
		dataDirectory = "Data";
		dataFile = "seeds.txt";
		with open(join(dataDirectory, dataFile), 'r') as file:
			for line in file:
				self.data.append([float(x) for x in line.strip().split()]);
		self.dataPoints = [None]*len(self.data);
	

	## Initialize the K centroids randomly				
	def initializeCentroids(self):
		centroids = rd.sample(range(len(self.data)), self.K);
		for x in centroids:
			self.dataCentroids.append(self.data[x]);
			self.centroids.append([]);
		

	## Update the distances of all points from the closest cluster
	def updateDistances(self):
		for x in range(len(self.centroids)):
			self.centroids[x] = [];
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
			self.centroids[minCentroid].append(x);
	
	## Update the centroids based on the points in the cluster
	def updateCentroids(self):
		for oldCentroidIdx in range(self.K):
			sumPoints = [0]*len(self.data[0]);
			for x in self.centroids[oldCentroidIdx]:
				sumPoints = list(map(add, sumPoints, self.data[x]));
			length = len(self.centroids[oldCentroidIdx]);
			newCentroid = [elem/length for elem in sumPoints];

			self.dataCentroids[oldCentroidIdx] = newCentroid;
	
	## Compute Sum of Squared Mean for all points
	def computeSSE(self):
		sumDistances = 0;
		for centroid in range(self.K):
			for point in self.centroids[centroid]:
				a = np.array(self.data[point]);
				b = np.array(self.dataCentroids[centroid]);
				distance = norm(a-b);
				sumDistances += distance**2;
		return sumDistances;


	## Fit the K-means algorithm to the given data
	def fit(self):
		self.initializeCentroids();
		sseVals = [];
		for x in range(self.maxIterations):	
			self.updateDistances();
			self.updateCentroids();
			sse = self.computeSSE();
			sseVals.append(sse);
			if len(sseVals) > 1 and abs(sseVals[len(sseVals)-1] - sseVals[len(sseVals)-2]) < self.epsilon:
				break;
		return sseVals[len(sseVals)-1];



def main():

	K = [3, 5, 7];	# all K values
	maxIterations = 100;	# maximum iterations
	epsilon = 0.001;	# stopping criteria 
	numRuns = 10;	# Number of runs to average
	
	for kval in K:
		
		## My Algorithm	
		kmeansMine = KMeansClustering(kval, maxIterations, epsilon, numRuns);
		
		# # sklearn library to confirm my results
		# kmeans = KMeans(kval, max_iter=100, tol=epsilon)
		# kmeans.fit(kmeansMine.data)
		# print(kmeans.inertia_);
		# [print(x) for x in kmeans.cluster_centers_];
	
if __name__ == '__main__':
	main();
