#pragma once
 struct ListNode {
     int val;
     ListNode *next;
     ListNode(int x) : val(x), next(nullptr) {}
 };

struct TreeNode {
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
};
//class Trie {
//	char v;
//	std::unordered_map<char, Trie*> next;
//	Trie(char _v) :v(_v) {};
//
//	static Trie *root;
//	static void add(std::string &word)
//	{
//		auto cur = Trie::root;
//		for (char c : word)
//		{
//			if (!cur->next.count(c))
//				cur->next[c] = new Trie(c);
//			cur = cur->next[c];
//		}
//	}
//	static bool find(std::string &word)
//	{
//		auto cur = Trie::root;
//		for (auto c : word)
//		{
//			if (!cur->next.count(c)) return false;
//			cur = cur->next[c];
//		}
//		return true;
//	}
//};