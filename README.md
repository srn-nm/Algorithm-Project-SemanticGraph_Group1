 Algorithm Design Course Project - Semantic Graph Search Algorithm

 In this project, we have designed and implemented a **Semantic Path Search System** that:
- Constructs a semantic graph from a set of input phrases
- Calculates edge weights based on **semantic similarity** between phrases
- Implements **BFS**, **Dijkstra**, and **A*** algorithms for finding the shortest path
- Uses **Language Models (Sentence Transformers)** to compute semantic similarity
- Provides complete **time and space complexity analysis**
- Evaluates performance on real and synthetic datasets



Team Members

Sarina Nasermoghaddasi - 40222083 -  Algorithm design, some part of the implementation , Complexity analysis 
Hananeh Ahmadi - 40247213 - some part of the implementation , Testing, Performance evaluation, Documentation, Demo 



------------------------------------------------



## Problem Definition

### Input:
- A set of `n` phrases (graph nodes)
- Source node `s` and target node `f`
- Language model (LLM) for semantic distance calculation

### Output:
- Shortest semantic path between `s` and `f`
- Path cost (total semantic distance)

### Formulation:
- Each node `v` represents a phrase
- Edge weight between nodes `u` and `v`:
  
  `Weight(u, v) = 1 - Similarity(u, v)
Team Members

Sarina Nasermoghaddasi - 40222083 -  Algorithm design, Core implementation, Complexity analysis 
Hananeh Ahmadi - 40247213 - Testing, Performance evaluation, Documentation, Demo 

`
  
- `Similarity(u, v) ∈ [0, 1]`
- Objective: Minimize the sum of weights along the path from `s` to `t`


### Characteristics:
- **Undirected** graph
- All weights are **positive**
- Graph can be **fully connected**, **threshold-based**, or **Top-K**



------------------------------------------------
  


## Algorithms

### 1. BFS Algorithm (Unweighted)
- Time Complexity: `O(V + E)`
- Space Complexity: `O(V)`
- Suitable for unweighted graphs

### 2. Dijkstra's Algorithm
- Time Complexity: `O((V+E) log V)`
- Space Complexity: `O(V)`
- **Selected Algorithm** due to:
  - Guaranteed shortest path
  - Efficient performance
  - No heuristic dependency


### 3. A* Algorithm
- Time Complexity: `O((V+E) log V)`
- Space Complexity: `O(V)`
- Heuristic function: `h(v) = 1 - Similarity(v, target)`


------------------------------------------------



## Installation

### Prerequisites
- Python 3.8+
- pip

### Installation Steps

1. **Clone the repository:**
git clone https://github.com/yourusername/Algorithm-Project-SemanticGraph-Group1.git
cd Algorithm-Project-SemanticGraph-Group1

2. **Create Virtual Environment:**
python -m venv venv
source venv/bin/activate  # Linux/Mac 
#or
venv\Scripts\activate  # Windows

3.Install dependencies:
bash
pip install -r requirements.txt

4.Run the demo:
bash
python interactive_demo.py


------------------------------------------------


### Installation Steps 


## 1. Run Main Program

# Build graph and search for path
python main.py --phrases data/phrases/tech_phrases.txt --build-graph --start "Artificial Intelligence" --end "Robotics" --algorithm dijkstra

# Compare algorithms
python main.py --phrases data/phrases/tech_phrases.txt --build-graph --start "Artificial Intelligence" --end "Robotics" --compare

# Performance evaluation
python main.py --phrases data/phrases/tech_phrases.txt --build-graph --evaluate

# Visualize graph
python main.py --phrases data/phrases/tech_phrases.txt --build-graph --visualize

## 2. Run Interactive Demo

python interactive_demo.py

## 3. Run Tests

python -m pytest tests/ -v
