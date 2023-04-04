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
        SolutionClimbStairs sc;
        int n = 5;
        auto r = sc.climbStairs(n);
        cout<<r<<endl;

    }

}
