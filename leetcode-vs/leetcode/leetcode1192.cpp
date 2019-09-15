#include <future>
#include <stack>
#include <unordered_map>
#include <unordered_set>
#include <bitset>
#include <set>
#include <queue>
#include <assert.h>
#include <iostream>
#include <vector>
#include <map>
#include <algorithm>
#include <thread>
#include <functional>
#include <mutex>
#include <string>
using namespace std;
typedef pair<int, int> ii;
const int N = 1e5 + 10;
vector<ii> a[N];
int dfn[N], low[N];
int bcc[N];

int stamp;
void DFS(int u, int up_edge, vector<vector<int>>& ret) {
	static stack<int> S;
	low[u] = dfn[u] = stamp++;
	S.push(u);
	for (auto& it : a[u]) {
		if (it.second == up_edge) continue;
		int v = it.first;
		if (dfn[v] == 0) {
			DFS(v, it.second, ret);
			low[u] = min(low[u], low[v]);
			if (low[v] > dfn[u]) {
				ret.push_back({ u, v });
				while (S.top() != v) {
					bcc[S.top()] = v;
					S.pop();
				}
				bcc[v] = v;
				S.pop();
			}
		}
		else {
			low[u] = min(low[u], dfn[v]);
		}
	}
}

class Solution {
public:
	//1192. Critical Connections in a Network
	vector<vector<int>> criticalConnections(int n, vector<vector<int>>& connections) {
		for (int i = 0; i < n; ++i) {
			a[i].clear();
			dfn[i] = 0;
		}
		for (int k = 0; k < connections.size(); ++k) {
			int x = connections[k][0];
			int y = connections[k][1];
			a[x].push_back({ y, k });
			a[y].push_back({ x, k });
		}
		stamp = 1;
		vector<vector<int>> ret;
		DFS(0, -1, ret);
		return ret;
	}
};

int main()
{
	return 0;
}