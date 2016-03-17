"""
jenks_caspall.py
Author : domlysz@gmail.com
Date : february 2016
License : GPL
Python 3

This is an attempt to implement Jenks-Caspall reiterative forcing algorithm.

This script correctly works as it but it's still in developement.
Future improvement will concerns classes design and clusters initialization method.

-------------------

The well know Natural Break classification can be computed through 2 algorithms:
* The Jenks-Caspall algorithm developed in 1971 is an empirical approach based on minimizing
the classification errors by moving observations between adjacent classes.
* The Fisher-Jenks algorithm, introduced to cartographers by Jenks in 1977, uses in contrast
a mathematical foundation, developing by Fisher in 1958, that guarantees an optimal solution.
It usually refers as "Jenks’s optimal method".

There are strong Python implementations of Jenks-Fisher available, but they can't process
lot of values in a reasonable time. The algorithm makes this somewhat inevitable : the number
of possible combination increases exponentially with the number of values, and find the optimal
partition will be increasingly slow. Usually softwares which uses Jenks-Fisher sample the data
befohand to maintain acceptable running time. Using an optimal method with only a sample of data
is a nonsense unless the sampling was taken very carefully. 


Jenks-Caspall algorithm was presented in the paper :
"Error on choroplethic maps: definition, measurement, reduction" (Jenks GF, Caspall FC, 1971) 

This algorithm is subdivised in two steps:
* Step 1 is similar to the well know K-means algorithm : starting with a giving partition, each
value is affected to its closest class mean. Then, means are updated and the process reiterates
until there is no more possible move.
* Step 2 try to optimize the partition by forcing values to their adjacent classes regardless
their distances to the classes means.

To find the best partition, Jenks introduced three errors criteria :
* Overview Accuracy Index (OAI) based on area of entities
* Tabular Accuracy Index (TAI) based on sum of absolute deviations about class means
* Boundary Accuracy Index (BAI) based on spatial neighborhood relationships of entities

In it's original paper, Jenks himself have focused on tabular error, primarily because of its simplicity.

Nowadays, another index named Goodness of Variance Fit (GVF) is more widelly used.
* GVF is based on sum of square deviation between values and mean.
* TAI is based on sum of absolute deviation between values and mean.
The best GVF will not necessarily be the best TAI and vice versa.

-------------------
Notes about K-means

K-means is generally used to classify multi dimensional data like 2D points. Traditionals K-means
implementations can be widely optimized to process 1 dimensional values which is completely ordered.


There have been many variations and studies on K-means algorithm over the past 50 years.
One of the most extensively discussed problem concerns the initialization method because
K-means performance is extremely sensitive to cluster center initialization. Bad initialization
can lead to poor convergence speed and bad overall clustering.

One the most recent proposal to this problem is the K-means++ algorithm (Arthur and Vassilvitskii 2007).
It help to initialize partition with centroids that are furthest away as possible from each other.
The first cluster center is chosen uniformly at random and each subsequent center is chosen with a 
probability proportional to its squared distance from the closest existing center.


Note that the random based initializations will give different results for multiple runs. It is a common 
practise to run k-means multiple times, and choose the better resulting partition.

#######################################

Original Jenks Caspall description:
	
	STEP 1: Reiterative Cycling
	a) The arrayed data are cut into arbitrary classes.
	b) Means for these classes are calculated.
	c) OAI, TAI, BAI, and MAI indices for the classes are calculated and stored in four
	separate locations along with the limits of each class.
	d) A new set of classes is created by grouping the intensity values to the nearest class mean.
	e) Means and accuracy indices for the new classes are calculated.
	f) Each new index is compared with its stored counterpart. If it is higher, the old
	set of classes is destroyed and the new set is stored in its place. If the new index is
	lower than the stored index, the new set is destroyed. 
	g) The process is continued until the class limits, means, and indices repeat themselves


	STEP 2: forcing cycling
	a) Starting with the classes obtained above, a new set is created by forcing one member
	into the class with the smallest mean, taking it from the class with the next smallest mean.
	(Remember that the data are arrayed and thus the smallest intensity in the second
	class is moved to the first class.)
	b) Calculate means and indices, and compare as before.
	c) If the TAI is increased move a second member into the class with the smallest
	mean. If the TAI is decreased discard the first set of force classes and return to the
	last set derived by reiteration.
	d) Force a member from the class with the third largest mean into the class with the
	second largest mean. Calculate and compare as above (2b and 2c). Continue if TAI
	is increased; if not, go to 2e.
	e) Forcing is continued until the class with the highest mean is reached. At this point
	upward forcing is initiated by moving a member from the class with the second
	largest mean into the class with the largest mean. Test and compare as before. This
	procedure is continued until the class with the smallest mean is reached.
	f) Continue the procedure with another downward forcing cycle.
	g) Continue the procedure with another upward forcing cycle.
	h) Continue the procedure by duplicating the set of classes with the highest TAI index
	(in storage) and reiterate it as in Step One.
	
-------------------
Slocum's description in his book "Thematic Cartography and Geographic Visualization":
	
	The Jenks—Caspall algorithm, developed by George Jenks and Fred Caspall (1971), is an empirical solution 
	to the problem of determining optimal classes. It minimizes the sum of absolute deviations about class means
	(as opposed to medians). The algorithm begins with an arbitrary set of classes (say, the quantiles classes),
	calculates a total map error, and attempts to reduce this error by moving observations between adjacent classes.

	Observations are moved using what Jenks and Caspall termed reiterative and forced cycling. In reiterative
	cycling, movements are accomplished by determining how close an observation is to the mean of another class.
	Movements based on the relation of observations to class means are repeated until no further reductions in
	total map error can be made. 

	In forced cycling, individual observations are moved into adjacent classes, regardless of the relation between
	the mean value of the class and the moved observation. After a movement, a test is made to determine whether
	any reduction in total map error has occurred. If error has been reduced, the new classiﬁcation is considered an
	improvement and the movement process continues in the same direction. Forcing is done in both directions 
	(from low to high classes and from high to low classes). At the Conclusion of forcing, the reiterative procedure
	described earlier is repeated to see whether any further reductions in error are possible. Although this approach
	does not guarantee an optimal solution, Jenks and Caspall indicated that they were "unable to generate, either
	purposefully or by accident, a better representation in any set of data.

"""

import random
#import math
#import numpy as np

#Globals variables used by both Clusters and Cluster classes
#they define how the variances will be computed
USE_MEAN = True #if False, use median
SQUARE_DEV = True #if False, use absolute deviation


def getMedian(values):
	n = len(values)
	idx = n // 2
	if (n % 2):# odd number
		return values[idx]
	else: # even number
		return (values[idx-1] + values[idx]) / 2


class Cluster():
	
	def __init__(self, data, i, j):
		'''
		We define a cluster with its first and last index
		these indices refer to the original dataset
		the last index refers to a value that is include in this cluster
		'''
		self.data = data
		self.i, self.j = i, j
	
	def __getitem__(self, idx):
		return self.values[idx]
	
	def __iter__(self):
		return iter(self.values)	

	def __str__(self):
		return str(self.values)
	
	def __len__(self):
		return len(self.values)
	
	@property
	def values(self):
		return self.data[self.i:self.j+1]

	@property
	def startValue(self):
		return self.data[self.i]

	@property
	def endValue(self):
		return self.data[self.j]

	@property
	def indices(self):
		return [self.i, self.j]

	@property
	def size(self):
		return len(self.values)

	@property
	def mean(self):
		values = self.values #for speed, avoid extracting values multiple time
		return sum(values) / len(values)
	
	@property
	def median(self):
		return getMedian(self.values)

	@property
	def sumSquareDev(self):
		m = self.mean #precompute the mean for avoids slowdown
		return sum( [(v - m)**2 for v in self.values] )
	
	@property
	def sumAbsDev(self):
		m = self.mean
		return sum( [abs(v - m) for v in self.values] )

	@property
	def variance(self):
		'''
		Within classe variance
		Result depends on how globals USE_MEAN and SQUARE_DEV are defined
		It will be :
		* the sum of square deviation from class values to class mean or median
		or
		* the sum of deviation from class values to class mean or median
		'''
		if USE_MEAN:
			m = self.mean
		else:
			m = self.median
		#
		if SQUARE_DEV:
			return sum( [(v - m)**2 for v in self.values] )
		else:
			return sum( [abs(v - m) for v in self.values] )
		



class Clusters():

	# helping functions for initialization

	def buildFromBreakPtsIdx(self, breakPointsIdx):
		n, k = len(self.data), len(breakPointsIdx)+1
		self.clusters = []
		for idx in range(k):
			if idx == 0:
				i = 0
			else:
				i = self.clusters[idx-1].j + 1
			#
			if idx == k - 1:
				j = n - 1
			else:
				# breakPoints[i] is the last value included to ith cluster
				j = breakPointsIdx[idx]
				if j == n-1: #last value is in breaks list
					j -= 1 #adjust so that the last value will be affected to last cluster
			#
			self.clusters.append(Cluster(self.data, i, j))

	def buildFromCentroidsIdx(self, centroidsIdx):
		k = len(centroidsIdx)
		breakPointsIdx = []
		for idx in range(k-1):
			i, j = centroidsIdx[idx], centroidsIdx[idx+1]
			m1, m2 = self.data[i], self.data[j]
			vIdx = i+1
			while True:
				v = self.data[vIdx]
				dst1 = abs(m1 - v)
				dst2 = abs(m2 - v)
				if dst1 > dst2:
					breakPointsIdx.append(vIdx-1)
					break
				else:
					vIdx +=1
		# build clusters with these breakpoints			
		self.buildFromBreakPtsIdx(breakPointsIdx)

	def values2Idx(self, values, findLastIdx=False):
		if not findLastIdx:
			return [self.data.index(v) for v in values]
		else:
			rvData = list(reversed(self.data))
			return [rvData.index(v) for v in values]



	def __init__(self, data, k, style='quantile'):
		'''
		Create k clusters with an initial classification
		'''

		self.data = data
		self.data.sort()
		n = len(self.data)

		#precompute data statistics
		self.dataMean = sum(self.data) / n
		self.dataMedian = getMedian(self.data)
		self.dataSumSquareDev = sum([(self.dataMean - v)**2 for v in self.data])
		self.dataSumAbsDev = sum([abs(self.dataMean - v) for v in self.data])
		self.dataSumSquareDevMedian = sum([(self.dataMedian - v)**2 for v in self.data])
		self.dataSumAbsDevMedian = sum([abs(self.dataMedian - v) for v in self.data])

		# a little hack to build clusters from breaks values
		# use it only for testing a predefined partition
		if type(k) is list:
			breakPoints = k
			breakPointsIdx = self.values2Idx(breakPoints)
			self.buildFromBreakPtsIdx(breakPointsIdx)
			return

		if not 0 < k < n:
			raise ValueError('Wrong expected number of classes')
		if style not in ['quantile', 'equal_interval', 'random', 'kpp', 'max']:
			print('Incorrect requested init style, use default style')
			style = 'quantile'

		#request only one classe
		if k == 1:
			self.clusters =  [ Cluster(self.data, 0, n-1) ]
					
		elif style == 'quantile':
			#quantile = number of value per clusters
			q = int(n//k) #floor division
			if q == 1:
				raise ValueError('Too many expected classes')	   
			# Make a list of Cluster object			 
			self.clusters = [Cluster(self.data, i, i+q-1) for i in range(0, q*k, q)]
			#  adjust the last index of the last cluster to the effective number of value
			self.clusters[-1].j = n-1
		
		elif style == 'equal_interval':
			mini, maxi = self.data[0], self.data[-1]
			delta =  maxi - mini
			interval = delta / k
			breakPointsIdx = []
			target = mini + interval
			for i, v in enumerate(self.data):
				if len(breakPointsIdx) == k-1:
					break
				if v > target:
					breakPointsIdx.append(i-1)
					target += interval
			#build clusters with these breakpoints
			self.buildFromBreakPtsIdx(breakPointsIdx)
		
		elif style == 'random':
			#generate random indices
			breakPointsIdx = random.sample(range(0, n-1), k)
			breakPointsIdx.sort()
			#build clusters with them as breakpoints
			self.buildFromBreakPtsIdx(breakPointsIdx)
		
		elif style == 'kpp':
			## kmeans++ initialization
			## this code is based on an example describe at 
			## http://stackoverflow.com/questions/5466323/how-exactly-does-k-means-work
			#
			# use kpp to init centroids or directly breaksPoints ?
			AS_CENTROIDS = True
			if AS_CENTROIDS:
				# to get k classes we need k centroids
				n_init = k
			else:
				# to get k classes we need k-1 breakpoints
				n_init = k - 1
			#
			# pick up a random value as first break value
			centroidsIdx = random.sample(range(n), 1)
			# n_init - 1 values remaining to find
			for cpt in range(n_init - 1):
				centroidsValues = [self.data[idx] for idx in centroidsIdx]
				# For each value compute the square distance to the nearest centroid
				dst = [min([(c-v)**2 for c in centroidsValues]) for v in self.data]
				# compute probability of each values
				sumDst = sum(dst)
				probs = [d/sumDst for d in dst]
				# compute the cumulative probability (range from 0 to 1)
				cumSum = 0 #cumulative sum
				cumProbs = [] #cumulative proba
				for p in probs:
					cumSum += p
					cumProbs.append(cumSum)
				# now try to find a new centroid 	
				find = False
				while not find:
					# generate a random probability (a float range from 0 to 1)
					r = random.random()
					# loop over the probability of each value...
					for i, p in enumerate(cumProbs):
						# ...and search for the first value with a probability higher than the random one
						if r < p:
							# add idx to our centroids list if it's not already there
							if i not in centroidsIdx:
								centroidsIdx.append(i)
								find = True
								break
			centroidsIdx.sort()
			# find the breakpoints corresponding to these centroids
			if AS_CENTROIDS:
				self.buildFromCentroidsIdx(centroidsIdx)
			else:
				# build our clusters with these centroids as breakpoints
				self.buildFromBreakPtsIdx(centroidsIdx)
			#print(centroidsIdx)
			#print(self.indices)
			
		elif style == 'max':
			# a simple method to get well spaced splits
			# because data is ordered, start with first and last values
			breakPointsIdx = [0, n-1]
			for cpt in range(k-1):
				breaksValues = [self.data[idx] for idx in breakPointsIdx]
				# For each value compute the square distance to the nearest centroid
				dst = [min([(c-v)**2 for c in breaksValues]) for v in self.data]
				# choose the value that has the greatest minimum-distance to the previously selected centers
				idx = dst.index(max(dst))
				breakPointsIdx.append(idx)
			#Exclude first and last values
			breakPointsIdx.sort()
			breakPointsIdx = breakPointsIdx[1:-1]
			# build our clusters with these centroids as breakpoints
			self.buildFromBreakPtsIdx(breakPointsIdx)




	def __str__(self):
		return str([c.values for c in self.clusters])
	
	def __getitem__(self, idx):
		return self.clusters[idx]

	def __setitem__(self, idx, cluster):
		if isinstance(cluster, Cluster):
			self.clusters[idx] = cluster
		else:
			raise ValueError('Requiere a Cluster instance not %s' %type(cluster))
		if not self.checkIntegrity():
			raise ValueError('Incorrect cluster definition for this position')

	def __iter__(self):
		return iter(self.clusters)

	def __len__(self):
		return len(self.clusters)
	
	@property
	def k(self):
		return len(self.clusters)
	
	@property
	def size(self):
		return len(self.clusters)

	@property
	def n(self):
		return len(self.data)
		
	@property
	def values(self):
		return [c.values for c in self.clusters]

	@property
	def indices(self):
		return [c.indices for c in self.clusters]

	@property
	def breaks(self):
		return [self.data[c.j] for c in self.clusters[:-1]]

	@property
	def breaksWithBounds(self):
		return [self.data[0]] + [self.data[c.j] for c in self.clusters]

	@property
	def dataVariance(self):
		if SQUARE_DEV:
			if USE_MEAN:
				return self.dataSumSquareDev
			else:
				return self.dataSumSquareDevMedian
		else:
			if USE_MEAN:
				return self.dataSumAbsDev
			else:
				return self.dataSumAbsDevMedian

	@property
	def withinVariances(self):
		return [c.variance for c in self.clusters]
			
	@property
	def sumWithinVariances(self):
		'''Sum of withim clusters sum square dev or sum abs dev'''
		return sum([c.variance for c in self.clusters])
	
	@property
	def betweenVariance(self):
		return self.dataVariance - self.sumWithinVariances

	@property
	def betweenVariance_calc(self):
		'''
		Between classes variance
		Sum of square deviation from classes means to total mean
		'''
		if SQUARE_DEV:
			if USE_MEAN:
				return sum([c.size * (self.dataMean - c.mean)**2 for c in self.clusters])
			else:
				return sum([c.size * (self.dataMedian - c.median)**2 for c in self.clusters])
		else:
			if USE_MEAN:
				return sum([c.size * abs(self.dataMean - c.mean) for c in self.clusters])
			else:
				return sum([c.size * abs(self.dataMedian - c.median) for c in self.clusters])

	@property
	def gvf(self):
		'''
		Goodness of Variance Fit
		ranges from 1 (perfect fit) to 0 (awful fit)
		if not SQUARE_DEV then this will be the Tabular Accuracy Index
		'''
		return 1 - (self.sumWithinVariances / self.dataVariance)


	def printStats(self):
		print('%i values, %i classes' %(self.n, self.k))
		#print("Breaks %s" %self.breaks)
		print("Breaks with bounds %s" %self.breaksWithBounds)
		#print("Data Variance %i" %self.dataVariance)
		print("Sum of within classes variances %i" %self.sumWithinVariances)
		print("Between classes variance %i" %self.betweenVariance)
		print("Goodness of variance fit %f" %self.gvf)  


	def checkIntegrity(self):
		# last index +1 of each cluster == start index of the next cluster
		return all( [c.j + 1 == self.clusters[idx+1].i for idx, c in enumerate(self.clusters[:-1])] )

	def moveForward(self, idx):
		'''
		Move last value of a given cluster index to its next cluster
		'''
		if idx == len(self.clusters) - 1:
			#last cluster, cannot move forward
			return False
		#if self.clusters[idx].i == self.clusters[idx].j:
		if self.clusters[idx].size == 1:
			# only one value remaining in this cluster
			# do not execute this move to avoid having an empty cluster
			return False
		#decrease right border index of current cluster
		self.clusters[idx].j -= 1
		#decrease left border index of the next cluster
		self.clusters[idx+1].i -= 1
		return True
		

	def moveBackward(self, idx):
		'''
		Move first value of a given cluster index to its previous cluster
		'''
		if idx == 0:
			#first cluster, cannot move backward
			return False
		if self.clusters[idx].size == 1:
			# only one value remaining in this cluster
			# do not execute this move to avoid having an empty cluster
			return False
		#increase left border index of the current cluster
		self.clusters[idx].i += 1
		#increase right border index of previous cluster
		self.clusters[idx-1].j += 1
		return True
		

	def save(self):
		self.previousIndices = self.indices

	def restore(self):
		for idx, c in enumerate(self.clusters):
			c.i, c.j = self.previousIndices[idx]

#-------------------



def kmeans1D(clusters, updateMeanAfterEachMove=False):
	nbIter = 0
	nbMoves = 0
	changeOccured = True
	while True:
		nbIter += 1
		nbMovesIter = 0
		changeOccured = False
		
		# save actual partition and sum of within variances
		sumWithinVariances = clusters.sumWithinVariances
		clusters.save()

		# if not updateMeanAfterEachMove, keep the same initial means during this iteration
		# means are re-computed only after all the data points have been assigned to their nearest centroids
		means = [c.mean for c in clusters]

		# for each border...
		k = len(clusters)
		for idx in range(k-1):
			c1, c2 = clusters[idx], clusters[idx+1]
			#m1, m2 = means[idx], means[idx+1]
			adjusted = False
				
			#try to adjust this border by moving forward (c1 -> c2)
			while True:		 
				breakValue = c1.endValue
				#get distance to means
				dst1 = abs(breakValue - means[idx])
				dst2 = abs(breakValue - means[idx+1])
				if dst1 > dst2: #this value will be better in c2		
					if clusters.moveForward(idx): #move is a success ...
						adjusted = True
						nbMovesIter += 1
						if updateMeanAfterEachMove: #always use an updated mean
							means[idx], means[idx+1] = c1.mean, c2.mean
					else:
						break
				else:
					break
				
			if not adjusted:
				# maybee we can do it backward (c1 <- c2)
				while True:
					breakValue = c2.startValue
					dst1 = abs(breakValue - means[idx])
					dst2 = abs(breakValue - means[idx+1])			   
					if dst2 > dst1: 
						if clusters.moveBackward(idx+1):
							adjusted = True
							nbMovesIter += 1
							if updateMeanAfterEachMove: #always use an updated mean
								means[idx], means[idx+1] = c1.mean, c2.mean
						else:
							break
					else:
						break

			if adjusted:
				changeOccured = True

		if not changeOccured:
			break
		elif clusters.sumWithinVariances > sumWithinVariances:
			#This new partition isn't better, so restaure the previous one and break the loop
			clusters.restore()
			break
		else:
			nbMoves += nbMovesIter
		
	return nbIter, nbMoves


#-------------------

def forceCycle(clusters):

	k = len(clusters)
	nbMoves = 0
	
	#store a list of variances (because we don't want to recompute all variance after each moves)
	#instead of GVF, we'll use variance (sum of squared or absolute deviation) as evaluation criterion
	#so we can just update and compare the sum of dev for the modified clusters
	#it will be faster than use GVF, and the results remaining the same
	sumdev = clusters.withinVariances

	# backward forcing cycle.
	# from low to high classes, move backward a value
	for idx in range(1,k):
		c1, c2 = clusters[idx-1], clusters[idx]
		while True:
			#
			previousSumDev = sumdev[idx-1] + sumdev[idx]
			#forcing first value of a given cluster index to its previous cluster
			moved = clusters.moveBackward(idx)
			if not moved:
				break
			newSumDev = [c1.variance, c2.variance]
			if sum(newSumDev) > previousSumDev:
				#undo last move and break loop
				clusters.moveForward(idx-1)
				break
			else:
				sumdev[idx-1], sumdev[idx] = newSumDev
				nbMoves +=1

	# forward forcing cycle
	# from high to low classes, move forward a value
	for idx in range(k-2, -1, -1):
		c1, c2 = clusters[idx], clusters[idx+1]
		while True:
			previousSumDev = sumdev[idx] + sumdev[idx+1]
			#forcing last value of a given cluster index to its next cluster
			moved = clusters.moveForward(idx)
			if not moved:
				break
			newSumDev = [c1.variance, c2.variance]
			if sum(newSumDev) > previousSumDev:
				#undo last move and break loop
				clusters.moveBackward(idx+1)
				break
			else:
				sumdev[idx], sumdev[idx+1] = newSumDev
				nbMoves +=1

	return nbMoves


#-------------------

def jenksCaspall(data, k, nbAttempt=1, initStyle='kmeanpp'):

	bestClusters = None
	for i in range(nbAttempt):
		
		print('Running Jenks-Caspall natural breaks...')
		print('**Attempt number %i' %(i+1))
		
		#kmean++ intit
		clusters = Clusters(data, k, initStyle)
		print('Step 1 : kmeans++ initalization, GVF = %f' %clusters.gvf)
		#print(clusters.breaks)
		
		#kmeans
		nbIter, nbMoves = kmeans1D(clusters)
		print('Step 2 : kmeans complete in %i iterations and %i moves, GVF = %f' %(nbIter, nbMoves, clusters.gvf))
		#print(clusters.breaks)

		#force cycle
		nbForceCycle = 0
		nbMovesAll = 0
		while True:
			nbForceCycle += 1   
			nbMoves = forceCycle(clusters)
			if not nbMoves:
				break
			else:
				nbMovesAll += nbMoves
		print('Step 3 : Forcing completed in %i cycles and %i moves, GVF = %f' %(nbForceCycle, nbMovesAll, clusters.gvf))
		#print(clusters.breaks)

		#Assign best partition
		if i == 0:
			bestClusters = clusters
		else:
			if clusters.gvf > bestClusters.gvf:
				bestClusters = clusters

	#Finish
	print('Jenks-Caspall competed!')
	bestClusters.printStats()
	
	return bestClusters



#############################"
if __name__ == '__main__':
	

	# This is the original dataset used by Jenks and Caspall in their paper
	# Jenks reach a TAI of 0.73416 with the following breaks [41.2, 58.5, 75.51, 100.10]
	
	data=[47.29, 15.57, 57.26, 28.71, 68.45, 55.44, 46.13, 116.40, 87.87, 84.84, 55.30, 57.49, 34.63, 31.19, 24.83, 85.41, 41.02, 67.04, 34.58, 32.14, 53.77, 92.45, 52.05, 17.82, 50.53, 50.12, 32.22, 60.66, 18.57, 66.32, 59.29, 119.90, 39.84, 41.20, 62.65, 31.78, 32.28, 96.37, 75.51, 50.52, 72.76, 47.21, 52.21, 71.61, 27.16, 155.30, 77.29, 62.06, 36.24, 33.26, 38.22, 50.11, 86.75, 45.40, 131.50, 60.27, 49.38, 57.22, 57.59, 80.45, 56.35, 68.19, 83.90, 111.80, 79.66, 64.28, 52.89, 68.25, 30.97, 33.83, 59.65, 57.40, 39.88, 54.27, 38.30, 33.82, 51.15, 100.10, 54.07, 31.28, 40.62, 28.20, 63.67, 32.69, 50.64, 39.72, 31.66, 37.57, 15.93, 96.78, 54.32, 71.05, 62.41, 58.50, 75.29, 23.42, 26.09, 40.36, 76.43, 52.47, 51.15, 73.52]

	print('---------------')

	k = 5
	nbAttempt = 4
	clusters = jenksCaspall(data, k, nbAttempt, initStyle='kpp')
	# >> results : TAI 0.735534, breaks [41.2, 60.66, 75.51, 100.1]
	
