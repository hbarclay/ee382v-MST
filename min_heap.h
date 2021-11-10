// A C++ program to demonstrate common Binary Heap Operations
#include<iostream>
#include<stdio.h>
#include<climits>
using namespace std;

class MinHeapNode
{
	public:
	int key;
	int val;
	MinHeapNode(int k, int v)
	{
		key=k;
		val=v;
	}
	MinHeapNode()
	{
		key=INT_MAX;
		val=INT_MAX;
	}
};

// Prototype of a utility function to swap two integers
void swap(MinHeapNode *x, MinHeapNode *y);

// A class for Min Heap
class MinHeap
{
	MinHeapNode *harr; // pointer to array of elements in heap
	int* pos;
	int capacity; // maximum possible size of min heap
	int heap_size; // Current number of elements in min heap
	public:
	// Constructor
	MinHeap(int capacity);

	// to heapify a subtree with the root at given index
	void MinHeapify(int );

	int parent(int i) { return (i-1)/2; }

	// to get index of left child of node at index i
	int left(int i) { return (2*i + 1); }

	// to get index of right child of node at index i
	int right(int i) { return (2*i + 2); }
	
	int size(){return heap_size;}
	bool empty(){ return (heap_size==0);  }
	// to extract the root which is the minimum element
	MinHeapNode extractMin();

	// Decreases key value of key at index i to new_val
	void decreaseKey(int i, int new_key);

	// Returns the minimum key (key at root) from min heap
	MinHeapNode getMin() { return harr[0]; }
	// return the node pointed by index
	MinHeapNode getNodeByIndex(int i){ 
		MinHeapNode bad(1234,1234);
		if(i>=heap_size){
			printf("getIndex Error! invalid Index\n");
			return bad;
		}
		else
			return  harr[i];
	}

	bool valExist(int v) {return ( (pos[v]!=INT_MAX)? true:false);}
	int getIndexByVal (int val) {return pos[val];}
	// Deletes a key stored at index i
	void deleteIndex(int i);

	// Inserts a new key 'k'
	void insert(int k, int v);
	//check if value v already exist in the heap, if not, insert, if yes, check if key k is lower than current key, if so, decrease it
	void insertOrDecrease(int k, int v);
};

// Constructor: Builds a heap from a given array a[] of given size
MinHeap::MinHeap(int cap)
{
	heap_size = 0;
	capacity = cap;
	harr = new MinHeapNode[cap];
	pos = new int[cap]; //save index position of node with particular val; assuming value is unqie
	for(int i =0 ; i<capacity;i++)
	{
		harr[i].key=INT_MAX;
		harr[i].val=INT_MAX;
		pos[i]=INT_MAX;
	}
}

// Inserts a new key 'k'
void MinHeap::insert(int k, int v)
{
	if(v>capacity-1)
	{
		cout <<"inserted value is larger than cap\n";
		return;
	}


	if (heap_size == capacity)
	{
		cout << "\nOverflow: Could not insert\n";
		return;
	}

	// First insert the new key at the end
	heap_size++;
	int i = heap_size - 1;
	harr[i].key = k;
	harr[i].val = v;
	pos[v]=i;

	// Fix the min heap property if it is violated
	while (i != 0 && harr[parent(i)].key > harr[i].key)
	{
		//swap position
		pos[ harr[i].val         ] = parent(i);
		pos[ harr[parent(i)].val ] = i;
		//actual swap
		swap(&harr[i], &harr[parent(i)]);
		i = parent(i);
	}
}

// Decreases value of key at index 'i' to new_val.  It is assumed that
// new_val is smaller than harr[i].
void MinHeap::decreaseKey(int i, int new_key)
{
	harr[i].key = new_key;
	while (i != 0 && harr[parent(i)].key > harr[i].key)
	{
		//swap position
		pos[ harr[i].val ] = parent(i);
		pos[ harr[parent(i)].val ] = i;
		//actual swap
		swap(&harr[i], &harr[parent(i)]);
		i = parent(i);
	}
}

void MinHeap::insertOrDecrease(int k, int v){
	if(valExist(v))
	{
		if(harr[getIndexByVal(v)].key>k)
			decreaseKey(getIndexByVal(v),k );
	}
	else
	{
		insert(k,v);
	}
}

// Method to remove minimum element (or root) from min heap
MinHeapNode MinHeap::extractMin()
{
	if (heap_size <= 0){
		printf("extractMin failed: heap size is zero\n");
		MinHeapNode bad(1234,1234);
		return bad;
	}
	if (heap_size == 1)
	{
		heap_size--;
		return harr[0];
	}

	// Store the minimum value, and remove it from heap
	MinHeapNode root = harr[0];
	harr[0] = harr[heap_size-1];
	
	//update position
	pos[root.val]=123456;
	pos[harr[0].val]=0;
	
	heap_size--;
	MinHeapify(0);

	return root;
}


// This function deletes key at index i. It first reduced value to minus
// infinite, then calls extractMin()
void MinHeap::deleteIndex(int i)
{
	pos[harr[i].val]=INT_MAX;
	decreaseKey(i, INT_MIN);
	extractMin();
}

// A recursive method to heapify a subtree with the root at given index
// This method assumes that the subtrees are already heapified
void MinHeap::MinHeapify(int i)
{
	int l = left(i);
	int r = right(i);
	int smallest = i;
	if (l < heap_size && harr[l].key < harr[i].key)
		smallest = l;
	if (r < heap_size && harr[r].key < harr[smallest].key)
		smallest = r;
	if (smallest != i)
	{
		pos[harr[i].val] = smallest;
		pos[harr[smallest].val] = i;
		swap(&harr[i], &harr[smallest]);
		MinHeapify(smallest);
	}
}

// A utility function to swap two elements
void swap(MinHeapNode *x, MinHeapNode *y)
{
	MinHeapNode temp = *x;
	*x = *y;
	*y = temp;
}
/*
// Driver program to test above functions
int main()
{
	MinHeap h(11);
	
	h.insert(7,5);
	h.insert(2,2);
	h.insert(3,6);
	h.insert(10,3);
	h.insert(9,1);
	h.insert(6,7);
	h.insert(1,4);
	
	for(int i=0;i < h.size(); i++)
	{
		printf("%d ", h.getNodeByIndex(i).key);
	}
	printf("\n");
	
	printf("val index test:\n");
	printf("val %d should have key %d, %d \n", 5,7,h.getNodeByIndex(h.getIndexByVal(5)).key);
	printf("val %d should have key %d, %d \n", 2,2,h.getNodeByIndex(h.getIndexByVal(2)).key);
	printf("val %d should have key %d, %d \n", 6,3,h.getNodeByIndex(h.getIndexByVal(6)).key);
	printf("val %d should have key %d, %d \n", 3,10,h.getNodeByIndex(h.getIndexByVal(3)).key);
	printf("val %d should have key %d, %d \n", 1,9,h.getNodeByIndex(h.getIndexByVal(1)).key);
	printf("val %d should have key %d, %d \n", 7,6,h.getNodeByIndex(h.getIndexByVal(7)).key);

	printf("min extract test:\n");
	int size = h.size();
	for(int i=0;i < size; i++)
	{
		printf("%d ", h.extractMin().key);
	}
	printf("\nshould be 1,2,3,6,7,9,10\n");

	
	return 0;
}
*/
