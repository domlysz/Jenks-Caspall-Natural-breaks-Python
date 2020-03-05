Python Jenks-Caspall Natural breaks
==========


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
beforehand to maintain acceptable running time. Using an optimal method with only a sample of data
is a nonsense unless the sampling was taken very carefully. 


Jenks-Caspall algorithm was presented in the paper :
"[Error on choroplethic maps: definition, measurement, reduction](https://www.jstor.org/stable/2562442?seq=1)" (Jenks GF, Caspall FC, 1971) 


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


