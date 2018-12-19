# lmu-artificial-intelligence
Homework assignments from CMSI485

## Homework 1
As defined in the problem specification "Your mission: implement A* graph search on a non-uniform cost maze pathfinding formulation of the traveling salesman problem!"
I implemented an informed search strategy making no assumptions that a solution exists, which returned the path with the least associated cost. This includes two Python classes that define the maze problem and the nodes within the problem. A* graph search is implemented to solve the given problem. 

## Homework 2
As defined in the problem specification "Your mission: implement a basic, slightly restricted propositional logic inference engine for use in our Maze Pitfall Problem!"
The first class defines the structure of the propositional logic clauses as a list of tuples. The other class implements a knowledge base that can be added to and respond to general queries.

## Homework 3
As defined in the problem specification "Your task: create a decision-network-based Ad Agent that will maximize the expected utility of a sale."
This problem required we determine the Bayesian network strcture using Tetrad, process a .csv file, and use Python's pomegranate library to query the decision network for the best combination of Ads.

## Homework 4
As defined in the problem specification "Your task: train and compare the performances of the supervised learning techniques we've discussed in the final third of this class on the datasets above! In particular, you will compare: Naive Bayes Classifiers, Decision Trees, Random Forests."
I needed to read, in great detail, the scikit, sklearn, numpy, and pandas documentation to get started on this machine learning problem. Learning to preprocess the data was the biggest challenge but was accomplished using imputing and encoding.
