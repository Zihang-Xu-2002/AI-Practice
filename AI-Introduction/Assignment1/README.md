#  Targets

完成5个函数

```C++
//Search
std::vector<Action> bfs(Problem& problem);
std::vector<Action> recursiveDLS(Node& node, Problem& problem, int limit);
std::vector<Action> aStar(Problem& problem, heuristicFunc heuristicFunc);

//Heuristic
int misplace(std::vector<int>& state, std::vector<int>& goalState);
int manhattan(std::vector<int>& state, std::vector<int>& goalState);
```



#  Action

用来描述方格的6种状态

```C++
enum class Action
{
	LEFT,
	UP,
	RIGHT,
	DOWN,
	FAILURE,
	CUTOFF
};
```

# Node

一个搜索结点的四个元素：

- n.state : 对应状态空间种的状态
- n.PARENT : 搜索树中的父结点
- n.ACTION : 父结点生成该结点时所采取的行动
- n.PATH-COST : 代价，一般用$g(n)$表示，指从初始结点到该结点的路径消耗

![image-20221012160917287](C:\Users\xzh\AppData\Roaming\Typora\typora-user-images\image-20221012160917287.png)

这里的parent指针很重要，求得问题的解之后用这个指针获得解路径。



```C++
//Node中的成员变量
Node* parent;
Action action;
int cost;
std::vector<int> state;

//function cmp : cost的比较
(Node*a,Node*b)
{
    return a->cost>b->cost
}


```



关于子结点的生成，算法伪代码如下

![image-20221012161011164](C:\Users\xzh\AppData\Roaming\Typora\typora-user-images\image-20221012161011164.png)



#  Problem

```C++
std::vector<int> initState;
std::vector<int> goalState;
std::set<Node*> nodePtrs;	
// 保存搜索过程中创建的节点的指针，用于内存管理，如果在代码中使用了new //Node，需要把该node的指针放到该变量中

std::vector<int> result(std::vector<int>state,Action action); //对应problem.RESULT函数

bool isGoal(std::vector<int>& state);//对应problem.GOAL-TEST函数
std::vector<Action> getValidActions(std::vector<int>& state);//对应problem.ACTIONS函数

bool isValidAction(std::vector<int>& state, Action action);	
bool isSolution(std::vector<Action>& solution, std::vector<int> initState);	
void updateInitState();		// 随机出一个的新的初始状态，并更新
void printState(std::vector<int>& state);
void printSolution(std::vector<Action>& solution);
void freeMemory();	// 用于内存管理，释放储存在nodePtrs中指针所指向的内存
void freeMemory(Node* node);	// 用于内存管理，在实现DLS需要调用
```



#  Heuristic

用来给$A^*$提供hCost

# Search  

```C++ 
Node childNode(Problem& problem, Node& parent, Action action, int cost);	// 对应CHILD-NODE函数
std::vector<Action> getSolution(Node node);		// 对应SOLUTION函数,到达目标之后获得路径
bool inSet(std::set<std::vector<int>>& collection, std::vector<int>& state);
bool inMap(std::map<std::vector<int>, Node*>& map, std::vector<int>& state);
// 把oldNode的数据替换为newNode中的数据，oldNode需要为frontier中的元素
void updateNode(std::priority_queue<Node*, std::vector<Node*>, Node::cmp>& frontier, Node* oldNode, Node* newNode);
// breadth first search 算法
std::vector<Action> bfs(Problem& problem);
// depth limit search 算法
std::vector<Action> recursiveDLS(Node& node, Problem& problem, int limit);
std::vector<Action> dls(Problem& problem, int limit);
std::vector<Action> dlsWrapper(Problem& problem);
// a* 算法
typedef int(*heuristicFunc)(std::vector<int>& state, std::vector<int>& goalState);
std::vector<Action> aStar(Problem& problem, heuristicFunc heuristicFunc);
std::vector<Action> aStarMisplace(Problem& problem);
std::vector<Action> aStarManhattan(Problem& problem);
```



##  BFS



##  recursiveDLS

![image-20221012163917772](C:\Users\xzh\AppData\Roaming\Typora\typora-user-images\image-20221012163917772.png)





##  $A^*$ 





