#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>


using namespace std;

class Solution {

public:
    int findContentChildren(vector<int>& g, vector<int>& s) {
//        g贪心指数  s： 饼干size
        long m = g.size();
        long n = s.size();

//        把饼干size和贪心指数从大到小排列O(nlogn)
        sort(g.begin(), g.end(), greater<int>());
        sort(s.begin(), s.end(), greater<int>());

        int si = 0, gi = 0;
        int res = 0;

//        贪心算法O(n)
        while(si < n && gi < m) {

            if(s[si] >= g[gi] ) {
                res ++;
                si ++;
                gi ++;
            }
            else{
                gi ++;
            }
        }

        return res;


    }

    int lcs(vector<char> s1, vector<char> s2) {
//        s1[0, m-1]  s2[0, n-1]
        long m = s1.size();
        long n = s2.size();

        int dp[m+1][n+1];

        for (int i = 0; i <= m; ++i) {
            for (int j = 0; j < n; ++j) {

                if(i == 0 || j == 0) {
                    dp[i][j] = 0;
                }
                else if(s1[i] == s2[j]) {
                    dp[i][j] = 1 + dp[i-1][j-1];
                }
                else {
                    dp[i][j] = max( dp[i-1][j], dp[i][j-1]);
                }
            }
        }

        return dp[m][n];

    }

private:

    int tryRob(vector<int>& nums, int index) {

        if(index >= nums.size())
            return 0;

        int res = 0;
        for (int i = index; i < nums.size() ; ++i) {
            res = max(res, (nums[i] + tryRob(nums, index + 2)));
        }

        return res;
    }




public:
    bool increasingTriplet(vector<int>& nums) {
        int c1 = INT_MAX, c2 = INT_MAX;
        int n = nums.size();

        for (int i = 0; i < n; ++i) {

            if(nums[i] <= c1) {
                c1 = nums[i];
            } else if(nums[i] <= c2) {
                c2 = nums[i];
            } else {
                return true;
            }
        }

        return false;

    }


    int lengthOfLIS(vector<int>& nums) {
        int len = nums.size();

        if(n == 0)
            return 0;

        vector<int> memo(n, 1);

        for (int i = 1; i < len; ++i) {

            for (int j = 0; j < i; ++j) {

                if(nums[i] > nums[j]) {
                    memo[i] = max(memo[i], 1+memo[j]);
                }
            }
        }

        int res = 1;
        for (int k = 0; k < len; ++k) {
            res = max(res, memo[k]);
        }

        return res;
    }

    bool canPartition(vector<int>& nums) {
        int n = nums.size();

        bool res;
        int sum = 0;
        for (int i = 0; i < n; ++i) {
            sum += nums[i];
        }

        if(sum%2 != 0)
            return false;

        int C = sum/2;
        bool dp[C+1][n+1];
        for (int i = 0; i < n; ++i) {
            dp[0][i] = true;
        }
        for (int i = 1; i <= C; ++i) {
            dp[i][0] = false;
        }

        for (int i = 1; i <= C; ++i) {

            for (int j = 0; j <= n ; ++j) {
                dp[i][j] = dp[i][j-1];
                if(i >= nums[j])
                    dp[i][j] = dp[i-nums[j-1]][j-1] || dp[i][j];
            }
        }

        return dp[C][n];
    }

    int coinChange(vector<int>& coins, int amount) {
        int n = coins.size();

        vector<int> dp(amount+1, INT_MAX);
        dp[0] = 0;

        for (int i = 1; i <= amount; ++i) {

            for (int j = 0; j < n; ++j) {
                if(i >= coins[j])
                    dp[i] = min(dp[i], dp[i-coins[j]] + 1);
            }
        }

        return dp[amount] > amount ? -1: dp[amount];



    }

    int rob(vector<int>& nums) {
         long n = nums.size();

        vector<int> res(n, 0);
        res[0] = nums[0];
        res[1] = max(nums[0], nums[1]);

        for(int i = 2; i < n; i++) {
            res[i] = max(res[i-1], nums[i] + res[i-2]);
        }

        return res[n-1];

    }

    int minPathSum(vector<vector<int>>& grid) {
        int m = grid.size();
        int n = grid[0].size();

        vector<vector<int>> memo(m, vector<int>(n, grid[0][0]));

        for (int i = 1; i < m; ++i) {
            memo[i][0] = memo[i-1][0] + grid[i][0];
        }

        for (int j = 1; j < n; ++j) {
            memo[0][j] = memo[0][j-1] + grid[0][j];
        }

        for (int i = 1; i < m; ++i) {

            for (int j = 1; j < n; ++j) {

                memo[i][j] = grid[i][j] + min(min(memo[i-1][j], memo[i][j-1]), memo[i-1][j-1]);
            }
        }

        return memo[m-1][n-1];
    }

    int minimumTotal(vector<vector<int>>& triangle) {
//        m:行，，n:列
        int m = triangle.size();
        int n = triangle[0].size();

        for (int i = m-2; i >= 0; i--) {

            for (int j = 0; j <= i; ++j) {

                if(triangle[i+1][j] > triangle[i+1][j+1]) {
                    triangle[i][j] += triangle[i+1][j];
                } else {
                    triangle[i][j] += triangle[i+1][j+1];
                }
            }

        }

        return triangle[0][0];
    }

    int maxProduct(vector<int>& nums) {
        int maxLocal = nums[0];
        int minLocal = nums[0];
        int res = nums[0];

        for (int i = 1; i < nums.size(); ++i) {
            int temp = maxLocal;
            maxLocal = max(max(nums[i], nums[i] * maxLocal), minLocal*nums[i]);
            minLocal = min(min(nums[i], nums[i] * temp), minLocal*nums[i]);

            res = max(res, maxLocal);
        }

        return res;
    }



    int rob(vector<int>& nums) {
        return tryRob(nums, 0);
    }

private:
    const string letterMap[10] {
            "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"
    };


    vector<bool > used;

    int d[4][2] = {{-1,0}, {0, 1}, {1, 0}, {0, -1}};
    int m, n;
    vector<vector<bool >> visited;

    vector<int > mem;

    int breakInter(int n) {

        if ( n == 1)
            return 1;

        if (mem[n] != -1)
            return mem[n];

        int res = -1;
        for (int i = 1; i < n; ++i) {

            res = max(res, max(i*breakInter(n-i), i*(n-i)));



        }

        mem[n] = res;

        return res;
    }

    bool inArea( int n) {

        for (int i = 1; i < (n+1)/2; ++i) {

            if (n == i*i) {
                return true;
            }
        }
        return false;
    }

    int dfs(int n, int num) {

        if (n == 1) {
            return 1;
        }

        int res = num;
        for (int i = 2; i < n; ++i) {

            if ( !inArea(n)) {
                num += dfs(n-i*i, num+1);
            } else {
                res = min(res, num+1);
            }



        }

        return res;
    }

public:


    int numSquares(int n) {
        return dfs(n, 0);

    }

    int integerBreak(int n) {

        mem = vector<int>(n+1, -1);
        return breakInter(n);
    }

    bool inArea(int x, int y) {
        return x >= 0 && x < m && y >= 0 && y < n;
    }

    bool searchWord( const vector<vector<char>>& board, const string& word, int index,
                     int startX, int startY) {

        if (index == word.size() - 1) {
            return board[startX][startY] == word[index];
        }

        if ( board[startX][startY] == word[index] ) {

            visited[startX][startY] = true;

            for (int i = 0; i < 4; ++i) {
                int newX = startX + d[i][0];
                int newY = startY + d[i][1];

                if(inArea(newX, newY) && !visited[newX][newY]
                        && searchWord(board, word, index+1, newX, newY)) {
                    return true;
                }
            }

            visited[startX][startY] = false;
        }

        return false;
    }

    bool exist(vector<vector<char>>& board, string word) {
        if (board.size() == 0) return false;
        m = board.size();
        assert( m > 0);

        n = board[0].size();

        visited = vector<vector<bool >>(m, vector<bool>( n, false));

        for (int i = 0; i < board.size(); ++i) {
            for (int j = 0; j < board[0].size(); ++j) {

                if ( searchWord(board, word, 0, i, j) )
                    return true;
            }
        }

        return false;
    }





    int sum(vector<int>& v) {
        int sum = 0;
        for (int i = 0; i < v.size(); ++i) {
            sum += v[i];
        }

        return sum;
    }

    void dfs(int target, vector<int>& candidates, vector<vector<int>>& res, vector<int>& temp, int begin ) {
        if (target == sum(temp)) {
            res.push_back(temp);
            return;
        }

        for (int i = begin; i < candidates.size(); ++i) {
            if (sum(temp) < target) {
                temp.push_back(candidates[i]);
                dfs(target, candidates, res, temp, i);
                temp.pop_back();
            }
        }

        return;
    }

    vector<vector<int>> combinationSum(vector<int>& candidates, int target) {
        vector<vector<int>> res;
        vector<int> temp;
        sort(candidates.begin(), candidates.end());

        if (candidates.size() == 0) return res;

        dfs(target, candidates, res, temp, 0);

        return  res;
    }



    void dfs(vector<int>& nums, vector<vector<int>>& res, int index, vector<int> temp) {

        if(index == nums.size() ) {
            res.push_back(temp);
            return;
        }

        for (int i = index; i < nums.size(); ++i) {

            if ( !used[i]) {
                temp.push_back(nums[i]);
                used[i] = true;
                dfs(nums, res, i+1, temp);
                temp.pop_back();
                used[i] = false;
            }
        }

        return;

    }

    void dfs(const string& s, int index, vector<string>& temp, vector<vector<string>>& res) {

        if (s.size() == index ) {
            res.push_back(temp);
            return;
        }

        for (int i = index; i < s.size(); ++i) {
            if (isPalindrome(s, index, i)) {

                temp.push_back(s.substr(index, i - index + 1));
                dfs(s, i+1, temp, res);
                temp.pop_back();
            }
        }
    }
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> res;
        vector<int > temp;

        if(nums.size() == 0) return res;

        used = vector<bool >(nums.size()+1, false);

        dfs(nums, res, 0, temp);

        return res;

    }

    bool isPalindrome(const string &s, int start, int end) {

        while (start <= end) {
            if(s[start ++] != s[end --])
                return false;


        }
        return true;
    }



};




int main() {
    std::cout << "Hello, World!" << std::endl;

    vector<vector<int>> memo = { {1, 2, 3},
                                 {4, 8, 2},
                                 {1, 5, 3} };

    return 0;
}