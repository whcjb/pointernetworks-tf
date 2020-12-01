=================
2.1.15 permutation sequence
类似动态规划
解释https://leetcode.com/problems/permutation-sequence/discuss/22507/%22Explain-like-I'm-five%22-Java-Solution-in-O(n)
代码https://leetcode.com/problems/permutation-sequence/discuss/22544/Easy-understand-Most-concise-C++-solution-minimal-memory-required/587132
如果n=4,序列={1,2,3,4}
列出所有的permutation
1+permutation of {2,3,4}
2+permutation of {1,3,4}
3+permutatuin of {1,2,4}
4+permutation of {1,2,3}
{1,2,3}三个数有3！个permutations可能，6个，所以上面总共有24个，
所以如果第k个，比如k=14应该在3+permutatuin of {1,2,4}中，
如果要找到这个需要(k-1)/6, 因为是0-index，所以减1
13 / (n - 1)! = 13 / 3! = 2, 使用n-1的原因是把第一个数单独提出来，看所有可能。1+permutation of{2,3,4},
而{1,2,3,4}在index=2的位置=3，所以first_num=3，
所以接下来就看3+{1,2,4}中{1,2,4}
1+{2,4}
2+{1,4}
4+{1,2}
此时k = 13 - 2 * (n-1)! = 1, 2表示跨过了两个

string getPermutation(int n, int k) {
	string a = "";
	for (int i = 1; i <= n; ++i) {
		a += to_string(i); 
	} // a = "12345...n"
	vector<int> fact(n+1); //动态规划初始值加1
	fact[0] = 1;
	for (int i = 1; i <= n; ++i) {
		fact[i] = fact[i-1] * i;
	}
	k--;
	string ans = "";
	for (int i = n - 1; i >=0; --i) {
		int index = k / fact[i]; // k = 13, 13 / (4 - 1)!, 计算第一个字符, index=2
		k %= fact[i]; // 用来计算后面的字符
		ans += a[index];
		//a.delete(index, index+1);
		a.erase(a.begin() + index); // "12345"删除2,剩余"1345..."
	}
	return ans;
}