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
    class Solution {
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

    class SolutionDk {
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


    void run(){
        //[[],[47,50],[33,41],[39,45],[33,42],[25,32],[26,35],[19,25],[3,8],[8,13],[18,27]]
        /*MyCalendar* mc = new MyCalendar();
        cout<<"result0: "<<mc->book(47,50)<<endl;
        cout<<"result1: "<<mc->book(33,41)<<endl;
        cout<<"result2: "<<mc->book(39,45)<<endl;*/

        //["foo", "bar", 1], ["foo", 1], ["foo", 3], ["foo", "bar2", 4], ["foo", 4], ["foo", 5]]
        //["love","high",10],["love","low",20],["love",5],["love",10],["love",15],["love",20],["love",25]]
//        TimeMap timeMap = *new TimeMap();
//        timeMap.set("foo", "bar", 1);  // 存储键 "foo" 和值 "bar" ，时间戳 timestamp = 1
//        cout<<"\nget: "<<timeMap.get("foo", 1);         // 返回 "bar"
//        cout<<"\nget: "<<timeMap.get("foo", 3);         // 返回 "bar", 因为在时间戳 3 和时间戳 2 处没有对应 "foo" 的值，所以唯一的值位于时间戳 1 处（即 "bar"） 。
//        timeMap.set("foo", "bar2", 4); // 存储键 "foo" 和值 "bar2" ，时间戳 timestamp = 4
//        cout<<"\nget: "<<timeMap.get("foo", 4);         // 返回 "bar2"
//        cout<<"\nget: "<<timeMap.get("foo", 5);         // 返回 "bar2"

//        timeMap.set("love", "high", 10);
//        timeMap.set("love", "low", 20);
//        cout<<" "<<timeMap.get("love", 5);
//        cout<<" "<<timeMap.get("love", 10);
//        cout<<" "<<timeMap.get("love", 15);
//        cout<<" "<<timeMap.get("love", 20);
//        cout<<" "<<timeMap.get("love", 25);

        //hand = [1,2,3,6,2,3,4,7,8], groupSize = 3
//        vector<int> vec{1,2,3,6,2,3,4,7,8};
//        cout<<LC::Solution::isNStraightHand(vec,3);

        //输入：buildings = [[2,9,10],[3,7,15],[5,12,12],[15,20,10],[19,24,8]]
        //输出：[[2,10],[3,15],[7,12],[12,0],[15,10],[20,8],[24,0]]
        vector<vector<int>> bds{{2,9,10},{3,7,15}};
//        vector<vector<int>> bds{{2,9,10},{3,7,15},{5,12,12},{15,20,10},{19,24,8}};
        SolutionSkyLine::getSkyline(bds);
    }

}
