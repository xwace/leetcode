//
// Created by star on 23-3-24.
//

#include <utility>
#include <set>
#include <iostream>
#include <unordered_set>
#include <map>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <iterator>
#include <list>
#include <queue>
#include <climits>
#include <stack>
#include "Leetcode.h"

using namespace std;
namespace LC{

    /**
     * description:二分查找日程安排的情况来检查新日程安排是否可以预订
     */
    class MyCalendar_ {
        std::set<std::pair<int,int>> booked;

    public:
        bool book(int start, int end) {
            auto it = booked.lower_bound({end, 0});
            if (it == booked.begin() || (--it)->second <= start) {
                booked.emplace(start, end);
                return true;
            }
            return false;
        }
    };

    class MyCalendar {
        //tree是叶子节点回溯到根节点经过的所有节点,lazy类似叶子节点;
        std::unordered_set<int> tree, lazy;

    public:
        bool query(int start, int end, int l, int r, int idx){

            //目标[start,end]与线段[l,r]无交集时候返回
            if (r < start or end < l) return false;

            //区间idx与[l,r]是绑定的
            if (lazy.count(idx))//如果该区间已被预订，则直接返回
                return true;
            if (start <= l and r <= end)//目标区间包含线段
                //tree是比lazy更大的区间;加快搜索过程
                //例如目标[2,3,4,5]切分为[2],tree[3,4,5]
                return tree.count(idx);//tree[3,4,5]存在,则一定有元素冲突;反之亦然

            auto mid =( l + r) / 2;

            return query(start, end, l, mid, 2 * idx) or query(start, end, mid + 1, r, 2 * idx + 1);
        }

        void update(int start, int end, int l, int r, int idx) {
            if (r < start or end < l) return;

            //目标区间相交二分的线段时,更新
            //if为目标区间包含线段树
            if (start <= l and r <= end) {
                tree.emplace(idx);
                lazy.emplace(idx);
            }else{
                auto mid = (l + r) / 2;
                update(start, end, l, mid, 2 * idx);
                update(start,end,mid + 1,r,2*idx+1);
                tree.emplace(idx);//idx对应的区间一定存在要插入的元素

                if (lazy.count(2 * idx)  and  lazy.count(2 * idx + 1)){
                    lazy.emplace(idx);
                }
            }
        }

        bool book(int start, int end) {
            if (query(start, end, 0, static_cast<int>(1e8), 1))
                return false;
            update(start, end, 0, static_cast<int>(1e8), 1);
            return true;
        }
    };


    //["foo","bar",1],["foo",1],["foo",3],["foo","bar2",4],["foo",4],["foo",5]]
    class TimeMap {
        std::map<string,map<int,string>>Map;
        std::unordered_map<string, std::vector<pair<int, string>>> m;
    public:
        TimeMap() = default;

        void set(string key, string value, int timestamp) {
            m[key].emplace_back(timestamp, value);
        }

        string get(string key, int timestamp) {
            auto &pairs = m[key];
            // 使用一个大于所有 value 的字符串，以确保在 pairs 中含有 timestamp 的情况下也返回大于 timestamp 的位置
            pair<int, string> p = {timestamp, string({127})};
            auto i = upper_bound(pairs.begin(), pairs.end(), p);
            if (i != pairs.begin()) {
                return (i - 1)->second;
            }
            return "";
        }
    };


    /**
     * description:输入：hand = [1,2,3,6,2,3,4,7,8], groupSize = 3
     * 输出：true Alice 手中的牌可以被重新排列为 [1,2,3]，[2,3,4]，[6,7,8]。
     * @date 3/28/20.57
     */
    class NStraightHandSolution {
    public:
        static bool isNStraightHand(vector<int> &hand, int groupSize) {
            int cnt = (int)hand.size() % groupSize;
            if (cnt!=0) return false;

            sort(hand.begin(),hand.end());

            std::unordered_map<int, int> map;
            for (auto &h: hand) map[h]++;

            for (auto &h: hand) {
                if (!map[h]) continue;

                for (int i = 0; i < groupSize; ++i) {
                    int num = h + i;
                    if (!map.count(num)) return false;
                    map[num]--;

                    if (map[num] == 0) map.erase(num);
                }

            }
            return true;
        }
    };

    /**天际线问题
     * description:输入：buildings = [[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]
     * 输出：[[2,10],[3,15],[7,12],[12,0],[15,10],[20,8],[24,0]]
     * @date 3/28 21.32
     */
    class SolutionSkyLine {
    public:
        static vector <vector<int>> getSkyline(vector <vector<int>> &buildings) {
            using PAIR = std::pair<int, int>;
            auto comp = [](const PAIR &a, const PAIR &b) { return a.second < b.second; };
            std::priority_queue<PAIR, vector<PAIR>, decltype(comp)> que(comp);

            vector<int> boundaries;
            for (auto &building: buildings) {
                boundaries.emplace_back(building[0]);
                boundaries.emplace_back(building[1]);
            }

            sort(boundaries.begin(), boundaries.end());

            vector<vector<int>> ret;
            int n = (int) buildings.size(), idx = 0;

            for (auto &boundary: boundaries) {
                cout<<"boud: "<<boundary<<endl;
                while (idx < n && buildings[idx][0] <= boundary) {
                    que.emplace(buildings[idx][1], buildings[idx][2]);
                    idx++;
                }

                cout<<"qtop: "<<que.top().first<<" "<<que.top().second<<endl;getchar();

                while (!que.empty() && que.top().first <= boundary) {
                    que.pop();
                }

                int maxn = que.empty() ? 0 : que.top().second;
                if (ret.empty() || maxn != ret.back()[1]) {
                    cout<<"boundary: "<<boundary<<" "<<maxn<<endl;
                    ret.emplace_back(boundary, maxn);
                }
            }

            return ret;
        }
    };

    //线段树离散化求解天际线问题
    class SegmentTree {
    public:
        struct node {
            int tag;
            int val;
        };

    private:
        using node_ptr = node*;
        node_ptr prefix, suffix;

        inline static int lowbit(int x) {
            return x & -x;
        }

    public:
        SegmentTree(size_t n) : prefix(new node[2 * n]), suffix(prefix + n) {}

        inline size_t size() const {
            return suffix - prefix;
        }

        inline void pushup(int p, int d, size_t s) {
            const int n = size();
            if (s == 0) return;
            int i = p + lowbit(s);
            int j = p - lowbit(s);
            for (; i <= n; i += lowbit(i))
                prefix[i].val = max(prefix[i].val, d);
            for (i -= lowbit(i); j > i; j -= lowbit(j))
                suffix[j].val = max(suffix[j].val, d);
        }

        inline void update(int l, int r, int d) {
            const int n = size();
            int i = l, j = r;
            if (l > 0)
                for (; i + lowbit(i) <= r; i += lowbit(i)) {
                    suffix[i].val = max(suffix[i].val, d);
                    suffix[i].tag = max(suffix[i].tag, d);
                }
            for (; j > i; j -= lowbit(j)) {
                prefix[j].val = max(prefix[j].val, d);
                prefix[j].tag = max(prefix[j].tag, d);
            }
            pushup(l, d, i - l);
            pushup(r, d, r - i);
        }

        inline int query(int p) const {
            const int n = size();
            int ans = 0;
            for (int i = p + 1; i <= n; i += lowbit(i))
                ans = max(ans, prefix[i].tag);
            for (int i = p; i > 0; i -= lowbit(i))
                ans = max(ans, suffix[i].tag);
            return ans;
        }
    };

    class SolutionSkyLine_{
    public:
        vector<vector<int>> getSkyline(vector<vector<int>>& buildings) {
            vector<int> order;
            for (const auto& e : buildings) {
                order.push_back(e[0]);
                order.push_back(e[1]);
            }
            auto first = order.begin();
            auto last = order.end();
            sort(first, last);
            last = order.erase(unique(first, last), last);
            const int n = order.size();
            SegmentTree tree(n);
            for (const auto& e : buildings) {
                const int l = lower_bound(first, last, e[0]) - first;
                const int r = lower_bound(first, last, e[1]) - first;
                tree.update(l, r, e[2]);
            }
            int prev = 0;
            vector<vector<int>> ans;
            for (int i = 0;i < n;++i) {
                const int cur = tree.query(i);
                if (cur != prev) ans.push_back({order[i], prev = cur});
            }
            return ans;
        }
    };

    /**
     * description:给出 nums = [1,3,-1,-3,5,3,6,7]，以及 k = 3。返回该滑动窗口的中位数数组 [1,-1,-1,3,5,6]。
     * @author oswin
     * @date 03/31 21.30
     */
    //fenwick树，树状数组
    class medianSlidingSolution1 {
    public:
        int maxn;
        vector<int> tree;
        multiset<int>window;

        int find(int x, vector<int> &nums) {
            return upper_bound(nums.begin(), nums.end(), x) - nums.begin();
        }

        //all_nums所有数值哈希化：[-1,3,6]->离散化转下标［4,3,1]->哈希化[1,0,1,1]
        //tree[i]表示前某几个哈希表值的和
        void update(int i, int v) {
            while (i < maxn) {
                tree[i] += v;
                i += (i & -i);
            }
        }

        //多个tree[i]相加，求i之前所有哈希值之和
        int query(int i) {
            int res = 0;
            while (i) {
                res += tree[i];
                i &= (i - 1);
            }
            return res;
        }

        int get_kth(int l, int r, int k) {
            while (l < r) {
                int mid = (l + r) >> 1;

                //二分查找哈希表中位数，对应滑动窗中值
                //query(mid)哈希表mid左侧个数
                if (query(mid) >= k) {
                    r = mid;
                } else {
                    l = mid + 1;
                }
            }

            return l;
        }

        vector<double> medianSlidingWindow(vector<int> &nums, int k) {
            vector<int> all_nums = nums;
            sort(all_nums.begin(), all_nums.end());
            all_nums.erase(unique(all_nums.begin(), all_nums.end()), all_nums.end());
            maxn = all_nums.size() + 5;
            tree.resize(maxn + 10);

            vector<double> res;
            for (int i = 0; i < nums.size(); ++i) {
                int j = find(nums[i], all_nums);
                update(j, 1);
                if (i >= k) {
                    j = find(nums[i - k], all_nums);
                    update(j, -1);
                }

                if (i < k - 1) continue;
                int m1 = get_kth(1, maxn, (k + 2) / 2);
                int m2 = (k & 1) ? m1 : get_kth(1, maxn, k / 2);
                res.push_back(((long long) all_nums[m1 - 1] + all_nums[m2 - 1]) * 0.5);
            }
            return res;
        }
    };

    //该算法超时
    class medianSlidingSolution2 {
    public:
        vector<double> medianSlidingWindow(vector<int> &nums, int k) {
            multiset<int> set;
            vector<double> res;

            for (int i = 0; i < nums.size(); ++i) {
                set.emplace(nums[i]);

                if (set.size() > k) {
                    auto it = set.lower_bound(nums[i - k]);
                    set.erase(it);
                }

                if (i < k) continue;
                res.emplace_back((double)*next(set.begin(), (k - 1) / 2) * 0.5 + (double)*next(set.begin(), k / 2) * 0.5);
            }

            return res;
        }
    };

    //双优先队列＋延迟删除
    //small保存大顶堆，小于等于top的入列,large大于top的入列
    class medianSlidingSolution {

    public:
        static vector<double> medianSlidingWindow(vector<int> &nums, int k) {
            DualHeap dh(k);
            for (int i = 0; i < k; ++i) {
                dh.insert(nums[i]);
            }

            vector<double> ans = {dh.getMedian()};
            for (int i = k; i < nums.size(); ++i) {
                dh.insert(nums[i]);
                dh.erase(nums[i - k]);
                ans.push_back(dh.getMedian());
            }

            copy(ans.begin(),ans.end(),ostream_iterator<int>(cout," "));
            return ans;
        }

        struct DualHeap {
            int k,smallSz{0},largeSz{0};
            unordered_map<int, int> delay;
            priority_queue<int, vector<int>, greater<>> large;
            priority_queue<int, vector<int>, less<>> small;

            explicit DualHeap(int k_) : k(k_) {}

            void insert(int i);
            void erase(int i);
            double getMedian() const;

            template<typename T>
            void prune(T &heap);
            void makeBalance();
        };
    };

    template<typename T>
    void medianSlidingSolution::DualHeap::prune(T &heap) {
        while (!heap.empty()) {
            int num = heap.top();
            if (delay.count(num)) {
                delay[num]--;
                if (!delay[num]) delay.erase(num);
                heap.pop();
            } else {
                break;
            }
        }
    }

    void medianSlidingSolution::DualHeap::makeBalance() {

        if (smallSz > largeSz + 1) {
            large.emplace(small.top());
            small.pop();
            smallSz--;
            largeSz++;
            prune(small);
        } else if (smallSz < largeSz) {
            small.emplace(large.top());
            large.pop();
            largeSz--;
            smallSz++;
            prune(large);
        }
    }

    void medianSlidingSolution::DualHeap::insert(int i) {

        if (small.empty() || i <= small.top()) {
            small.emplace(i);
            smallSz++;
        } else {
            large.emplace(i);
            largeSz++;
        }
        makeBalance();
    }

    double medianSlidingSolution::DualHeap::getMedian() const {
        return k % 2 ? small.top() : (double) (small.top() + large.top()) * 0.5;
    }

    void medianSlidingSolution::DualHeap::erase(int i) {

        delay[i]++;
        if (i <= small.top()) {
            smallSz--;
            if (i == small.top()) {
                prune(small);
            }
        } else {
            largeSz--;
            if (i == large.top()) {
                prune(large);
            }
        }
        makeBalance();
    }

    /**
     * description:返回一个新数组 counts;有该性质： counts[i] 的值是  nums[i] 右侧小于 nums[i] 的元素的数量。
     * @date 4/4 10.08
     */
    class SolutionCountSmallerNumbers {
        vector<int> tree;
        int maxn;
    public:

        void update(int i, int v) {
            while (i < maxn) {
                tree[i] += v;
                i += (i & -i);
            }
        }

        int query(int i) {
            int sum = 0;
            while (i > 0) {
                sum += tree[i];
                i -= (i & -i);
            }

            return sum;
        }

        vector<int> countSmaller(vector<int> &nums) {
            int len = (int) nums.size();
            maxn = len + 10;
            tree.resize(maxn, 0);
            std::set<int> set;
            vector<int> result(len,0);
            multiset<int>mset;

            for (int i = 0; i < len; ++i) {
                set.emplace(nums[i]);
            }

            for (int i = len - 1; i >= 0; --i) {
                //方法１multiset超时
                //mset.emplace(nums[i]);
                //auto it = mset.lower_bound(nums[i]);
                //int dis = (int)distance(mset.begin(),it);
                //result[i] = dis;

                //方法２　fenwick树
                auto it = set.lower_bound(nums[i]);
                int j = (int)distance(set.begin(), it);//这里耗时长；建议改成vector查找代替set
                update(j+1, 1);
                int r = query(j);
                result[i] = (r);
            }

            //[10,27,10,35,12,22,28,8,19,2,12,2,9,6,12,5,17,9,19,12,14,6,12,5,12,3,0,10,0,7,8,4,0,0,4,3,2,0,1,0]
            copy(result.begin(),result.end(),ostream_iterator<int>(cout," "));
            return result;
        }
    };

    /**
     * description:最长连续递增序列是 [1,3,5], 长度为3。
     */
    class SolutionLengthOfLCIS {
    public:
        int findLengthOfLCIS(vector<int>& nums) {
            int k{0};
            int max{-1};
            for (int i = 1; i < nums.size(); ++i) {
                k++;
                if (nums[i] >= nums[i - 1]){
                    if (k>max) max = k;
                    k = 0;
                }
            }

            return max;
        }
    };

    /**
     * description:栅格地图总共有多少条不同的路径？
     */
    class SolutionUniquePaths {
    public:
        int uniquePaths(int m, int n) {
            vector<vector<int>>dp(m,vector<int>(n,0));
            for (int i = 0; i < n; ++i) {
                dp[0][i] = 1;
            }

            for (int i = 0; i < m; ++i) {
                dp[i][0] = 1;
            }

            for (int i = 1; i < m; ++i) {
                for (int j = 1; j < n; ++j) {
                    dp[i][j] = dp[i][j-1] + dp[i-1][j];
                }
            }

            return dp[m-1][n-1];
        }
    };

    /**
     * description:爬楼顶
     */
    class SolutionClimbStairs {
    public:
        int climbStairs(int n) {
            if (n == 1) return 1;
            if (n == 2) return 2;
            if (n == 3) return 3;

            vector<int>mem(100,0);
            mem[3] = 3;
            mem[2] = 2;
            for(int i = 4;i<=n;i++){
                mem[i] = mem[i-1] + mem[i-2];
            }
            return mem[n];
        }
    };

    /**
     * description:找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。
     */
    class SolutionMinPathSum {
    public:
        int minPathSum(vector<vector<int>>& grid) {
            int m = (int)grid.size();
            int n = (int)grid[0].size();
            vector<vector<int>> dp(m,vector<int>(n));

            for (int i = 1; i < n; ++i) {
                grid[0][i] += grid[0][i-1];
            }

            for (int i = 1; i < m; ++i) {
                grid[i][0] += grid[i-1][0];
            }

            for (int i = 1; i < m; ++i) {
                for (int j = 1; j < n; ++j) {
                    grid[i][j] += std::min(grid[i-1][j],grid[i][j-1]);
                }
            }

            cout<<"min: "<<grid[m-1][n-1]<<endl;
            return grid[m-1][n-1];
        }
    };

    /**
     * description:answer[i] % answer[j] == 0 ，或answer[j] % answer[i] == 0
     */
    class SolutionDivisibleSubset {
    public:
        vector<int> largestDivisibleSubset(vector<int> &nums) {
            int len = (int) nums.size();
            vector<int> sortNums = nums;
            sort(sortNums.begin(), sortNums.end());
            vector<int> result;
            vector<int> dp(len);
            dp[0] = 1;

            for (int i = 1; i < len; ++i) {
                for (int j = 0; j <= i - 1; ++j) {
                    if (sortNums[i] % sortNums[j] == 0) {
                        dp[i] = std::max(dp[i],dp[j]+1);
                    }
                }

                if (dp[i] == 0) dp[i] = 1;
            }

            auto it = max_element(dp.begin(), dp.end());
            int id{0};
            for (int i = (int)dp.size()-1; i >=0; --i) {
                if (*it == dp[i]) {
                    id = i;
                    break;
                }
            }

            cout<<"it: "<<*it<<endl;
            cout<<"dpmax: "<<dp[id]<<endl;
            cout<<"id: "<<sortNums[id]<<endl;

            vector<int>chain;
            std::stack<pair<int,int>>stack;
            stack.emplace(dp[id],sortNums[id]);
            while (!stack.empty()){
                auto pr = stack.top();
                stack.pop();
                chain.emplace_back(pr.second);

                if (pr.first == 1){
                    swap(result,chain);
                    break;
                }

                int flag{0};
                for (int i = len - 1; i >= 0; --i) {
                    if (dp[i] == pr.first - 1){
                        if (pr.second%sortNums[i]==0){
                            stack.emplace(dp[i],sortNums[i]);
                            flag++;
                        }
                    }
                }
            }

            vector<int>temp;
            for (int k = (int)result.size()-1; k>=0; --k) {
                auto iter = find(nums.begin(),nums.end(),result[k]) - nums.begin();
                temp.emplace_back(nums[iter]);
            }

            swap(temp,result);

            copy(result.begin(),result.end(),ostream_iterator<int>(cout," "));
            return result;
        }
    };

    /**
     * description:最长递增子序列是 [2,3,7,101]，因此长度为 4 。
     * @date 04/05 18.48
     */
    class SolutionLengthOfLIS {
    public:
        int lengthOfLIS(vector<int>& nums) {
            int len = (int)nums.size();
            vector<int>dp(len,1);

            //方法１　动态规划
            for (int i = 1; i < len; ++i) {
                //int max = INT_MIN;
                for (int j = 0; j < i; ++j) {
                    if (nums[i] > nums[j])
                    {
                        //dp[i]作为存放max的临时变量
                        dp[i] = std::max(dp[i],dp[j]+1);
                        //max = std::max(max,dp[j] + 1);
                        //dp[i] = max;
                    }
                }
            }

            //方法２贪心算法＋二分查找替换；
            //vector<int>d;
            //for (auto num:nums) {
            //    auto it = lower_bound(d.begin(),d.end(),num);
            //    if (it==d.end()) d.emplace_back(num);
            //    else swap(*it,num);
            //}

            auto it = max_element(dp.begin(),dp.end());
            return *it;
        }
    };

    /**
     * description:当另一个信封的宽度和高度都比这个信封大的时候，这个信封就可以放进另一个信封里
     * 最多能有多少个 信封能组成一组“俄罗斯套娃”信封
     */
    class SolutionMaxEnvelopes {
    public:
        int maxEnvelopes(vector<vector<int>>& envelopes) {
            int len = (int)envelopes.size();
            vector<int>dp(len,1);
            auto comp = [](vector<int>a,vector<int>b){
                if(a[0]==b[0]) return a[1]>b[1];//相同的只取最大值，否则会多选;[4,5][4,6]会选到两个；
                //[4,6] [4,5]则只取[4,6]
                else return a[0]<b[0];
            };

            sort(envelopes.begin(),envelopes.end(),comp);

            //{5,4},{6,4},{6,7},{2,3}
            for (int i = 1; i < len; ++i) {
                for (int j = 0; j < i; ++j) {
                    if (envelopes[j][0] < envelopes[i][0] and envelopes[j][1] < envelopes[i][1])
                    {
                        dp[i] = std::max(dp[i],dp[j]+1);
                    }
                }

            }

            //方法２　二分查找替换
            //vector<int>d;
            //for (int i = 0; i < len; ++i) {
            //    auto it = lower_bound(d.begin(),d.end(),envelopes[i][1]);
            //    if (it == d.end()) d.emplace_back(envelopes[i][1]);
            //    else swap(*it,envelopes[i][1]);//*it= envelopes[i][1]
            //}
            //return (int)d.size();

            copy(dp.begin(),dp.end(),ostream_iterator<int>(cout," "));
            return *max_element(dp.begin(),dp.end());
        }
    };

    /**
     * description:粉刷完所有房子最少的花费成本。
     * 将 0 号房子粉刷成蓝色，1 号房子粉刷成绿色，2 号房子粉刷成蓝色。
     * 最少花费: 2 + 5 + 3 = 10。
     * @date 4/6 10.24
     */
    class SolutionMinCost {
    public:
        int minCost(vector<vector<int>>& costs) {
            int n = (int)costs.size(), house, color;
            vector<vector<int>> houseCost(n,vector<int>(3,0));
            houseCost[0] = costs[0];

            for (int i = 1; i < n; ++i) {
                houseCost[i][0] = costs[i][0] + std::min(houseCost[i-1][1],houseCost[i-1][2]);
                houseCost[i][1] = costs[i][1] + std::min(houseCost[i-1][0],houseCost[i-1][2]);
                houseCost[i][2] = costs[i][2] + std::min(houseCost[i-1][0],houseCost[i-1][1]);
            }

            int cost = std::min( std::min(houseCost[n-1][0],houseCost[n-1][1]),std::min(houseCost[n-1][1],houseCost[n-1][2]));
            return cost;
        }
    };

    /**
     * description:解释：在第 2 天（股票价格 = 1）的时候买入，在第 5 天（股票价格 = 6）的时候卖出，最大利润 = 6-1 = 5
     * 注意利润不能是 7-1 = 6, 因为卖出价格需要大于买入价格；同时，你不能在买入前卖出股票
     */
    class SolutionMaxProfit {
    public:
        int maxProfit(vector<int>& prices) {

            int minprice = INT_MAX, maxprofit = 0;
            for (int price: prices) {
                maxprofit = max(maxprofit, price - minprice);
                minprice = min(price, minprice);
                cout<<"max: "<<maxprofit<<" "<<minprice<<endl;
            }
            return maxprofit;

            //优先队列top保存最小值;max保存最大差值
            //priority_queue<int,vector<int>,greater<>>que;
            //que.emplace(prices[0]);
            //int max=0;
            //for (int i = 1; i < (int)prices.size(); ++i) {
            //    max = std::max(max,prices[i] - que.top());
            //    que.emplace(prices[i]);
            //}
            //return max;
        }
    };

    /**
     * description:输入：nums = [2,3,1,1,4]
     * 可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。
     */
    class SolutionCanJump {
    public:
        bool canJump(vector<int> &nums) {
            int len = (int) nums.size();
            if (len == 1) return true;

            int maxLoc{0};
            for (int i = 0; i < len; ++i) {
                if (nums[i]+i > maxLoc){
                    maxLoc = nums[i]+i;
                }
                if (i==maxLoc and i < len-1) return false;
            }
            return true;
        }
    };

    /**
     * description:跳到最后一个位置的最小跳跃数是 2。
     * @date 14.44-16.13
     */
    class SolutionJump {
    public:
        int jump(vector<int> nums) {
            int max_far = 0;// 目前能跳到的最远位置
            int step = 0;   // 跳跃次数
            int end = 0;    // 上次跳跃可达范围右边界（下次的最右起跳点）
            for (int i = 0; i < nums.size() - 1; i++)
            {
                max_far = std::max(max_far, i + nums[i]);
                // 到达上次跳跃能到达的右边界了
                if (i == end)
                {
                    end = max_far;  // 目前能跳到的最远位置变成了下次起跳位置的有边界
                    step++;         // 进入下一次跳跃
                }
            }
            return step;
        }
    };

    /**
     * description:给你一个字符串 s，请你将 s 分割成一些子串，使每个子串都是 回文串 。返回 s 所有可能的分割方案
     * 16.36
     */
    class SolutionPartition {
    private:
        vector<vector<int>> f;
        vector<vector<string>> ret;
        vector<string> ans;
        int n;

    public:
        void dfs(const string& s, int i) {
            if (i == n) {
                ret.push_back(ans);
                return;
            }

            for (int j = i; j < n; ++j) {
                if (f[i][j]) {//剪枝
                    ans.push_back(s.substr(i, j - i + 1));
                    dfs(s, j + 1);
                    ans.pop_back();
                }
            }
        }

        vector<vector<string>> partition(string s) {
            n = (int)s.size();
            f.assign(n, vector<int>(n, true));

            for (int i = n - 1; i >= 0; --i) {
                for (int j = i + 1; j < n; ++j) {
                    f[i][j] = (s[i] == s[j]) && f[i + 1][j - 1];
                }
            }

            dfs(s, 0);

            return ret;
        }
    };

    /**
     * description:将 s 分割成一些子串，使每个子串都是回文
     * 返回符合要求的 最少分割次数 。
     * @date 4/8 10.28
     */
    class SolutionMinCut {
    public:
        int minCut(string s) {
            int n = (int)s.size();
            vector<vector<int>> g(n, vector<int>(n, true));

            for (int i = n - 1; i >= 0; --i) {
                for (int j = i + 1; j < n; ++j) {
                    g[i][j] = (s[i] == s[j]) && g[i + 1][j - 1];
                }
            }

            vector<int> f(n, INT_MAX);
            for (int i = 0; i < n; ++i) {
                if (g[0][i]) {
                    f[i] = 0;
                }
                else {
                    for (int j = 0; j < i; ++j) {
                        if (g[j + 1][i]) {
                            f[i] = min(f[i], f[j] + 1);
                        }
                    }
                }
            }
            return f[n - 1];
        }
    };

    /**
     * description: 戳气球求所能获得硬币的最大数量
     * @date 4/9 15.07
     */
    class SolutionMaxCoins {
        vector<vector<int>> rec;
        vector<int> val;
    public:
        int solve(int left, int right) {

            if (left >= right - 1) {
                return 0;
            }
            if (rec[left][right] != -1) {
                return rec[left][right];
            }

            for (int i = left + 1; i < right; i++) {
                int sum = val[left] * val[i] * val[right];
                sum += solve(left, i) + solve(i, right);
                rec[left][right] = max(rec[left][right], sum);
            }
            return rec[left][right];
        }

        int maxCoins(const vector<int>& nums) {
            //方法１逐个插入气球
            int n = (int)nums.size();
            val.resize(n + 2);
            for (int i = 1; i <= n; i++) {
                val[i] = nums[i - 1];
            }
            val[0] = val[n + 1] = 1;
            rec.resize(n + 2, vector<int>(n + 2, -1));
            return solve(0, n + 1);

            //方法２动态规划
            vector<int> balloons(n + 2, 1);
            copy(nums.begin(), nums.end(), balloons.begin() + 1);
            vector<vector<int>> dp(n + 2, vector<int>(n + 2, 0));

            for (int i = n-1; i >= 0; --i) {
                for (int j = i + 2; j < n + 2; j++) {
                    for (int k = i + 1; k < j; ++k) {
                        //最后一个气球为Ｋ
                        dp[i][j] = std::max(dp[i][j], dp[i][k] + dp[k][j] + balloons[i] * balloons[k] * balloons[j]);
                    }
                }
            }

            return dp[0][n+1];

        }
    };

    /**
     * description:输入：text1 = "abcde", text2 = "ace"
     * 输出：3
     * 解释：最长公共子序列是 "ace" ，它的长度为 3 。
     * @date 4/10 1153
     */
    class SolutionCommonSubsequence {
    public:
        int longestCommonSubsequence(string text1, string text2) {
            int n1 = (int) text1.length();
            int n2 = (int) text2.length();

            vector<vector<int>> dp(n1 + 1, vector<int>(n2 + 1, 0));

            for (int i = 0; i < n1; ++i) {
                char c1 = text1[i];
                for (int j = 0; j < n2; ++j) {
                    char c2 = text2[j];
                    if (c1 == c2) {
                        dp[i + 1][j + 1] = dp[i][j] + 1;
                    } else {
                        dp[i + 1][j + 1] = std::max(dp[i][j + 1], dp[i + 1][j]);
                    }
                }
            }
            return dp[n1][n2];
        }
    };

    /**
     * description:最长重复子串
     */
    string dpLRS(string s)
    {
        int n = (int)s.length();
        vector<vector<int>>dp(n+1,vector<int>(n+1,0));
        int max{-1};
        string ans;

        for (int i = 0; i < n; ++i) {
            for (int j = i+1; j < n; ++j) {
                if (s[i] == s[j])
                {
                    dp[i+1][j+1] = dp[i][j]+1;
                }else{
                    dp[i + 1][j + 1] = std::max(dp[i][j + 1], dp[i + 1][j]);
                }

                max = std::max(max,dp[i+1][j+1]);
            }
        }

        cout<<ans<<endl;
        cout<<"longstring: "<<dp[n][n]<<endl;getchar();
    }

    /**
     * description:地下城游戏
     * @date 4/18 1446
     */
    class SolutionMinimumHP {
    public:
        int calculateMinimumHP(vector <vector<int>> dungeon) {
            int m = (int) dungeon.size();
            int n = (int) dungeon[0].size();

            vector<vector<int>> dp(m + 1, vector<int>(n + 1, INT_MAX));
            dp[m][n - 1] = dp[m - 1][n] = 1;

            for (int i = m - 1; i >= 0; --i) {
                for (int j = n - 1; j >= 0; --j) {
                    dp[i][j] = max(min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j], 1);
                }
            }

            return dp[0][0];
        }
    };

    /**
     * description:给你两个字符串 s 和 t ，统计并返回在 s 的 子序列 中 t 出现的个数。
     * @date 4/19 1018
     */
    class SolutionNumDistinct {
    public:
        int numDistinct(string s, string t) {
            int m = (int) s.length();
            int n = (int) t.length();

            if (m < n) return 0;

            vector<vector<size_t>> dp(m + 1, vector<size_t>(n + 1, 0));
            for (int i = 0; i < m + 1; ++i) {
                dp[i][0] = 1;
            }

            for (int i = 1; i <= m; ++i) {
                for (int j = 1; j <= n; ++j) {
                    if (s[i] == t[j]) {
                        dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
                    } else {
                        dp[i][j] = dp[i - 1][j];
                    }
                }
            }

            return (int)dp[m][n];
        }
    };

    class NumDecodings {
    public:
        static int numDecodings(const string& s) {
            int n = (int) s.length();
            vector<int> dp(n, 0);
            dp[0] = (s[0] == '0') ? 0 : 1;

            for (int i = 1; i < n; ++i) {
                auto val = stoi(string{s[i - 1]}) * 10 + stoi(string{s[i]});
                //单独一位0不能解码;s[i]值至少从1开始
                if (s[i] != '0') {
                    dp[i] += dp[i - 1];
                }

                if (val <= 26 and s[i - 1] != '0') {
                    //(i > 1)加哨兵 dp[-1]代替这行: if (i == 1) dp[i] += 1;
                    if (i == 1) dp[i] += 1;
                    else dp[i] += dp[i - 2];
                }
            }

            return dp[n-1];
        }

    };

    /**
     * description:输入：s = "1*" 输出：18
     * 这一条编码消息可以表示 "11"、"12"、"13"、"14"、"15"、"16"、"17"、"18" 或 "19" 中的任意一条。
     * @date 4/22 1708
     */
    class NumDecodings1 {
        static constexpr int mod = 1000000007;
    public:
        static int numDecodings(string s){
            int n = (int) s.length();
            vector<size_t> dp(n+1, 0);
            dp[0] = 1;

            auto oneChar = [](char c){
                if (c == '0') return 0;
                if (c == '*') return 9;
                return 1;
            };

            auto twoChars = [](char a, char b) {
                if (a == '*' and b == '*') return 15;
                if (a == '*') return b <= '6' ? 2 : 1;

                if (b == '*') {
                    if (a == '1') return 9;
                    if (a == '2') return 6;
                    return 0;
                }

                return (int)(((a - '0') * 10 + b - '0') <= 26 && a != '0');
            };

            int a = 0, b = 1, c = 0;
            for (int i = 1; i <= n; ++i) {

                auto c1 =  s[i-1];
                dp[i] = (dp[i] + oneChar(c1)*dp[i - 1])%mod;

                if (i>1){
                    auto c2 = s[i-2];
                    dp[i] = (dp[i] + twoChars(c2,c1)*dp[i - 2])%mod;
                }
            }

            return (int)dp[n];
        }
    };

    /**
     * description:给定两个字符串s1 和 s2，返回 使两个字符串相等所需删除字符的 ASCII 值的最小和
     * 输入: s1 = "sea", s2 = "eat"输出: 231
     * 解释: 在 "sea" 中删除 "s" 并将 "s" 的值(115)加入总和。在 "eat" 中删除 "t" 并将 116 加入总和
     * 结束时，两个字符串相等，115 + 116 = 231 就是符合条件的最小和。
     * @date 4/23 1638
     */
    class MinimumDeleteSum {
    public:
        static int minimumDeleteSum(string s1, string s2) {
            int m = (int) s1.length();
            int n = (int) s2.length();

            vector<vector<int>> dp(m + 1, vector<int>(n + 1, 0));

            for (int i = 1; i < m + 1; ++i) {
                dp[i][0] = dp[i - 1][0] + s1[i - 1];
            }

            for (int i = 1; i < n + 1; ++i) {
                dp[0][i] = dp[0][i - 1] + s2[i - 1];
            }

            for (int i = 1; i < m + 1; ++i) {
                for (int j = 1; j < n + 1; ++j) {
                    if (s1[i - 1] == s2[j - 1]) {
                        dp[i][j] = dp[i - 1][j - 1];
                    } else {
                        dp[i][j] = std::min(dp[i - 1][j] + s1[i - 1], dp[i][j - 1] + s2[j - 1]);
                    }
                }
            }

            return dp[m][n];
        }
    };

    /**
     * description:在一个由 '0' 和 '1' 组成的二维矩阵内，找到只包含 '1' 的最大正方形，并返回其面积。
     * 输入：matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
     * 输出：4
     * @date 4/26 1142
     */
    class MaximalSquare {
    public:
        static int maximalSquare(vector <vector<char>> &matrix) {
            int m = (int) matrix.size();
            int n = (int) matrix[0].size();

            vector<vector<int>> dp(m, vector<int>(n));

            int max{-1};
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (i == 0 or j == 0) {
                        dp[i][j] = matrix[i][j] - '0';
                        max = std::max(max, dp[i][j]);
                        continue;
                    }

                    dp[i][j] = matrix[i][j] == '1' ? std::min(min(dp[i][j - 1], dp[i - 1][j]), dp[i - 1][j - 1]) + 1 : 0;
                    max = std::max(max, dp[i][j]);
                }
            }

            return max * max;
        }
    };

    /**
     * 输入：matrix =
     * [[0,1,1,1],
     * [1,1,1,1],
     * [0,1,1,1]]
     * 输出：15
     * 边长为 1 的正方形有 10 个;边长为 2 的正方形有 4 个;边长为 3 的正方形有 1 个
     * 正方形的总数 = 10 + 4 + 1 = 15*/
    class CountSquares {
    public:
        static int countSquares(vector<vector<int>>& matrix) {
            int m = (int) matrix.size();
            int n = (int) matrix[0].size();
            int sum{0};;

            vector<vector<int>> dp(m, vector<int>(n));
            for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                    if (matrix[i][j] == 1) {
                        if (i == 0 or j == 0) dp[i][j] = 1;
                        else {
                            dp[i][j] = min(min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1;
                        }
                        sum += dp[i][j];
                    }
                }
            }

            return sum;
        }
    };

     /* description:输入：[1,2,3,1]
      * [2,7,9,3,1]
     * 输出：4
     * 偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。
     * 偷窃到的最高金额 = 1 + 3 = 4 。
     * @date 4 28 /1635*/


    void run(){
        //[[],[47,50],[33,41],[39,45],[33,42],[25,32],[26,35],[19,25],[3,8],[8,13],[18,27]]
        /*MyCalendar* mc = new MyCalendar();
        cout<<"result0: "<<mc->book(47,50)<<endl;
        cout<<"result1: "<<mc->book(33,41)<<endl;
        cout<<"result2: "<<mc->book(39,45)<<endl;*/

        //["foo", "bar", 1], ["foo", 1], ["foo", 3], ["foo", "bar2", 4], ["foo", 4], ["foo", 5]]
        //["love","high",10],["love","low",20],["love",5],["love",10],["love",15],["love",20],["love",25]]
        //TimeMap timeMap = *new TimeMap();
        //timeMap.set("foo", "bar", 1);  // 存储键 "foo" 和值 "bar" ，时间戳 timestamp = 1
        //cout<<"\nget: "<<timeMap.get("foo", 1);         // 返回 "bar"
        //cout<<"\nget: "<<timeMap.get("foo", 3);         // 返回 "bar", 因为在时间戳 3 和时间戳 2 处没有对应 "foo" 的值，所以唯一的值位于时间戳 1 处（即 "bar"） 。
        //timeMap.set("foo", "bar2", 4); // 存储键 "foo" 和值 "bar2" ，时间戳 timestamp = 4
        //cout<<"\nget: "<<timeMap.get("foo", 4);         // 返回 "bar2"
        //cout<<"\nget: "<<timeMap.get("foo", 5);         // 返回 "bar2"

        //timeMap.set("love", "high", 10);
        //timeMap.set("love", "low", 20);
        //cout<<" "<<timeMap.get("love", 5);
        //cout<<" "<<timeMap.get("love", 10);
        //cout<<" "<<timeMap.get("love", 15);
        //cout<<" "<<timeMap.get("love", 20);
        //cout<<" "<<timeMap.get("love", 25);

        //hand = [1,2,3,6,2,3,4,7,8], groupSize = 3
        //vector<int> vec{1,2,3,6,2,3,4,7,8};
        //cout<<LC::Solution::isNStraightHand(vec,3);

        //输入：buildings = [[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]
        //输出：[[2,10],[3,15],[7,12],[12,0],[15,10],[20,8],[24,0]]
        //vector<vector<int>> bds{{2,9,10},{3,7,15}};
        //vector<vector<int>> bds{{2,9,10},{3,7,15},{5,12,12},{15,20,10},{19,24,8}};
        //SolutionSkyLine::getSkyline(bds);

        //给出 nums = [1,3,-1,-3,5,3,6,7]，以及 k = 3
        //返回该滑动窗口的中位数数组 [1,-1,-1,3,5,6]
        //vector<int> medians{1,3,-1,-3,-5,-3,6,7};
        //int k = 6;
        //medianSlidingSolution ms;
        //LC::medianSlidingSolution::medianSlidingWindow(medians,k);

        //输入：nums = [5,2,6,1]
        //输出：[2,1,1,0]
        //vector<int> vec{83,51,98,69,81,32,78,28,94,13,2,97,3,76,99,51,9,21,84,66,65,36,100,41};
        //SolutionCountSmallerNumbers ss;
        //ss.countSmaller(vec);

        //输入：nums = [1,3,5,4,7]
        //输出：3
        //SolutionLengthOfLCIS sl;
        //vector<int>vec{3,1,2};
        //cout<<"lenght: "<<sl.findLengthOfLCIS(vec);

        //一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为 “Start” ）。
        //机器人试图达到网格的右下角总共有多少条不同的路径？（在下图中标记为 “Finish” ）
        //int m = 23, n = 12;
        //SolutionUniquePaths su;
        //su.uniquePaths(m,n);

        //输入：n = 2
        //输出：2
        //SolutionClimbStairs sc;
        //int n = 5;
        //auto r = sc.climbStairs(n);
        //cout<<r<<endl;

        //输入：grid = [[1,3,1],[1,5,1],[4,2,1]]
        //输出：7 因为路径 1→3→1→1→1 的总和最小
        //SolutionMinPathSum sm;
        //vector<vector<int>> grid{{1,3,1},{1,5,1},{4,2,1}};
        //sm.minPathSum(grid);


        //输入：nums = [1,2,3]
        //输出：[1,2]
        //SolutionDivisibleSubset sd;
        //vector<int> vec{5,9,18,54,108,540,90,180,360,720};
        //sd.largestDivisibleSubset(vec);

        //输入：nums = [10,9,2,5,3,7,101,18]
        //输出：4
        //SolutionLengthOfLIS sl;
        //vector<int>nums{0,1,0,3,2,3};{0,8,4,12,2}
        //cout<<"nums: "<<sl.lengthOfLIS(nums);

        //输入：envelopes = [[5,4],[6,4],[6,7],[2,3]]
        //输出：3
        //vector<vector<int>>envelopes{{5,4},{6,4},{6,7},{2,3}};
        //SolutionMaxEnvelopes sm;
        //cout<<"nums: "<<sm.maxEnvelopes(envelopes);

        //输入: [[17,2,17],[16,16,5],[14,3,19]]
        //输出: 10
        //vector<vector<int>>costs{{17,2,17},{16,16,5},{14,3,19}};
        //SolutionMinCost sm;
        //cout<<"mincost: "<<sm.minCost(costs);

        //输入：[7,1,5,3,6,4]
        //输出：5
        //SolutionMaxProfit sm;
        //vector<int>prices{7,1,5,3,6,4};
        //cout<<"profit: "<<sm.maxProfit(prices);

        //SolutionCanJump sj;
        //vector<int>js{2,0,0};
        //cout<<"can jump: "<<sj.canJump(js);

        //输入: nums = [2,3,1,1,4]
        //输出: 2
        //SolutionJump sj;
        //cout<<"jump: "<<sj.jump({7,0,9,6,9,6,1,7,9,0,1,2,9,0,3});

        //输入：s = "aab"
        //输出：[["a","a","b"],["aa","b"]]
        //SolutionPartition sp;
        //sp.partition(string{"aabbcc"});

        //SolutionMinCut sm;
        //cout<<"mincut: "<<sm.minCut(string{"aabbc"});

        //输入：nums = [3,1,5,8]
        //输出：167
        //nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
        //coins =  3*1*5    +   3*5*8   +  1*3*8  + 1*8*1 = 167
        //戳气球
        //SolutionMaxCoins sm;
        //cout<<"maxCoins: "<<sm.maxCoins(vector<int>{3,1,5,8});

        //最长公共子序列
        //SolutionCommonSubsequence sc;
        //cout<<"long subs: "<<sc.longestCommonSubsequence("abcded","ace");

        //输入：dungeon = [[-2,-3,3],[-5,-10,1],[10,30,-5]]
        //输出：7
       //SolutionMinimumHP sm;
       //cout<<"minHp: "<<sm.calculateMinimumHP(vector<vector<int>>{{{-2,-3,3}, {-5,-10,1},{10,30,-5}}})<<endl;

        //输入：s = "rabbbit", t = "rabbit"
        //输出：3
        //SolutionNumDistinct sn;
        //auto s = "rabbbit", t = "rabbit";
        //cout<<"minDistinct: "<<sn.numDistinct(s,t);

        //输入：s = "12"
        //输出：2
        //解释：它可以解码为 "AB"（1 2）或者 "L"（12）。
        //cout<<"num decode: "<<NumDecodings1::numDecodings("104")<<endl;

        //auto s = "delete", t = "leet";
        //cout<<"min delte: "<<MinimumDeleteSum::minimumDeleteSum(s,t)<<endl;

        //vector<vector<char>>matrix{{'1','0','1','0','0'},{'1','0','1','1','1'},{'1','1','1','1','1'},{'1','0','0','1','0'}};
        //vector<vector<int>>matrixi{{1,0,1,0,0},{1,0,1,1,1},{1,1,1,1,1},{1,0,0,1,0}};
        //cout<<"maxsquare: "<<MaximalSquare::maximalSquare(matrix)<<endl;
        //cout<<"maxsquare: "<<CountSquares::countSquares(matrixi)<<endl;

    }

}
