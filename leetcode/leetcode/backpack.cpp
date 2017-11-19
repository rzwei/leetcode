#include "stdafx.h"
#include <iostream>
#include <vector>
#include <algorithm>
using namespace std;
class backpack {
public:
	int zeroOnePack(vector<int>worth, vector<int>volume, int Volume)
	{
		int length = worth.size();
		vector<int>dp(length + 1);
		for (int i = 1; i <= length; i++)
			for (int v = Volume; v >= volume[i]; v--)
				dp[v] = max(dp[v], dp[v - volume[i]] + worth[i]);
		return dp[length];
	}
	int zeroOnePack2(vector<int>worth, vector<int>volume, int Volume)
	{
		int length = worth.size();
		vector<int>dp(length + 1, 0x80000000);
		dp[0] = 0;
		for (int i = 1; i < length; i++)
			for (int v = Volume; i >= volume[i]; i--)
				dp[v] = max(dp[v], dp[v - volume[i]] + worth[i]);
		return dp[length];
	}
	int zeroOneCompetePack(vector<int>worth, vector<int>volume, int Volume)
	{
		int length = worth.size();
		vector<int>dp(length + 1);
		for (int i = 1; i < length; i++)
			for (int v = volume[i]; v < length; v++)
				dp[v] = max(dp[v], dp[v - volume[i]] + worth[i]);
		return dp[length];

	}
};