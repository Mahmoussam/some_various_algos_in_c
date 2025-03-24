/*
#ifdef ONLINE_JUDGE
        freopen("input.txt", "r", stdin);
        freopen("output.txt", "w", stdout);
#endif
    
*/
// 0x3f3f3f3f
//random  xor hashing
const ll SEED=chrono::steady_clock::now().time_since_epoch().count();
mt19937_64 rng(SEED);
inline ll rnd(ll l=0,ll r=INF)
{return uniform_int_distribution<ll>(l,r)(rng);}
struct Hash{
    ll h1 , h2;
    // XOR operator: Combines two Hash objects (bitwise XOR for h1 and h2)
    Hash operator^(const Hash& other) const {
        return {h1 ^ other.h1, h2 ^ other.h2};
    }

    // Equality check: Compares both h1 and h2
    bool operator==(const Hash& other) const {
        return (h1 == other.h1) && (h2 == other.h2);
    }
    Hash& operator^=(const Hash& other) {
        h1 ^= other.h1;
        h2 ^= other.h2;
        return *this;
    }
    bool operator<(const Hash& other) const {
        return (h1 < other.h1) || (h1 == other.h1 && h2 < other.h2);
    }

    // Greater-than operator (optional, for completeness)
    bool operator>(const Hash& other) const {
        return (h1 > other.h1) || (h1 == other.h1 && h2 > other.h2);
    }
};

//lower_bound
vector<int> vec = {1 , 2 , 3 ,4 ,  4, 4 ,6 ,10};
cout<< (int)(lower_bound(vec.begin(),vec.end() , 4) - vec.begin()) <<endl;

#include <bits/stdc++.h>

using namespace std;
#define ll long long
int Nums[5];
void initiaition();
void prime();
struct st{
    int x,y;
    st(int xx,int yy){
        x=xx;
        y=yy;
    }
    st(){}
};
struct DSU{
    vector<int> sizes,parents;
    int components=0;
    int mx_s,mx_n;
    int len;
    DSU(int n){
        sizes=vector<int>(n,1);
        parents=vector<int>(n);
        for(int i=0;i<n;i++)parents[i]=i;
        components=n;
        mx_s=0;
        
    }
    int find_set(int x){//path compression
        if(x==parents[x])return x;
        return parents[x]=find_set(parents[x]);
    }
    void link(int x,int y){
        if(sizes[x]<sizes[y])swap(x,y);
        parents[y]=x;
        sizes[x]+=sizes[y];
        sizes[y]=0;
        
        if(sizes[x]>mx_s){
            mx_s=sizes[x];
            mx_n=x;
        }
    }
    bool union_sets(int x,int y){
        x=find_set(x);y=find_set(y);
        if(x!=y){
            link(x,y);
            components--;
        }
        return x!=y;
    }
    
};
/*ll szs[OL],pars[OL];
void make_set(int n){
    for(int i=0;i<n;i++){
        szs[i]=1;
        pars[i]=i;
    }
}
int find(int x){
    if(x==pars[x])return x;
    return pars[x]=find(pars[x]);
}
bool merge(int x,int y){
    x=find(x);y=find(y);
    if(x!=y){
        if(szs[x]<szs[y])swap(x,y);
        szs[x]+=szs[y];
        szs[y]=0;
        pars[y]=x;
    }
    return x!=y;
}*/
int main(){
    /*int* ptr = Nums;
    initiaition();
    prime();
    */
    //cout<<*(ptr + 1);
    
    {
    vector<int>v={1,2,3,3,3,4,5,6};
    auto it=upper_bound(v.begin(),v.end(),3);
    cout<<(int)(it-v.begin())<<endl;
    }
    /*auto it=upper_bound(v.begin(),v.end(),3);
    cout<<(int)(it-v.begin());*/
    /*vector<int>temp;
    temp.push_back(v[0]);
    for(int i=1;i<v.size();i++){
        if(v[i]>=temp.back())temp.push_back(v[i]);
        else{
            auto it=upper_bound(temp.begin(),temp.end(),v[i]);
            *it=v[i];
        }
    }
    for(auto x:temp)cout<<x<<" ";*/
    /*priority_queue<int,vector<int>,greater<int>> pq;
    pq.push(5);
    pq.push(10);
    cout<<pq.top()<<endl;*/
    
}
// Fenwick BIT
ll BIT[11]={0};
ll n[]={0,1,2,3,4,5,6,7,8,9,10};
ll query(int i){
    ll sum=0;
    while(i>0){
        sum+=BIT[i];
        i-=(i & -i);//remove last bit
    }
    return sum;
}
void update(int i,ll k){
    while(i<11){
        BIT[i]+=k;
        i+= (i& -i);
    }
}
//BIT INVERSION
ll query(int i,vector<int>&BIT){
    ll sum=0;
    while(i < BIT.size()){
        sum+=BIT[i];
        i+=(i & -i);//remove last bit
    }
    return sum;
}
void update(int i,ll k , vector<int>&BIT){
    while(i > 0){
        BIT[i]+=k;
        i-= (i& -i);
    }
}
ll calc_inversions(const vector<int>&v , int n){
    vector<int>BIT(OL , 0);
    ll z = 0;
    for(int i = n - 1;i >= 0; i--){
        z += query(v[i] , BIT);
        update(v[i] , 1 , BIT);
    }
    return z;
}
void prime(){

    for(int i = 0; i<5 ; i++){
        bool isPrime = true;
        for(int j = 2; j<Nums[i] ; j++){
            if(Nums[i] % j == 0)
                isPrime = false;
            
        }
        if (Nums[i] == 0 || Nums[i] == 1)
                isPrime = false;
        if(isPrime == true)
            cout<<Nums[i]<<" is a prime number."<<endl;
        

    }


}
//CHT
//x:start pos ,m,b==>y=m*x+b
struct seg{long double x;ll m,b;};
vector<seg>hull;
ll query(ll x){
    if(!hull.size())return 0;
    seg s= *--upper_bound(hull.begin(),hull.end(),x,[](ll a,seg b){return a<b.x;});
    return s.b+s.m * x;
}
void insert(ll m,ll b){
    while(hull.size()){
        seg s=hull.back();
        if(s.b+s.m*s.x>b+m*s.x){
            if(s.m-m)hull.push_back({(b-s.b)/(long double)(s.m-m),m,b});
            return;
        }
        hull.pop_back();
    }
    hull = {{LLONG_MIN,m,b}};
}
struct SegTree{
    vector<int>tree;
    int n;
    void build(int oldn,int *arr){
        if(__builtin_popcount(oldn)!=1)
            this->n=1<<(__lg(oldn)+1);
        else 
            this->n=oldn;
        tree.resize(this->n << 1,1e9);
        for(int i=0;i<oldn;i++)
            tree[i+this->n]=arr[i];
        for(int i=this->n-1;i>=0;i--){
            tree[i]=min(tree[i<<1],tree[i<<1 | 1]);
        }
    }
    int query(int ql,int qr,int k,int sl,int sr){
        //if seg inside ,return seg
        if(sl>=ql&&sr<=qr)return tree[k];
        if(ql>sr||qr<sl)return 1e9;
        int mid=(sl+sr)/2;
        return min(query(ql,qr,k<<1,sl,mid),query(ql,qr,k<<1 |1 ,mid+1,sr));
    }
    //point update
    void update(int ql,int qr,int v,int k,int sl,int sr){
        //if seg inside ,return seg
        if(sl>=ql&&sr<=qr){
            tree[k]=v;
            return;
        }
        if(ql>sr||qr<sl)return;
        int mid=(sl+sr)/2;
        update(ql,qr,v,k<<1,sl,mid);
        update(ql,qr,v,k<<1 |1 ,mid+1,sr);
        //re calculate
        tree[k]=min(tree[k<<1],tree[k<<1|1]);
    }
    void show(){
        for(auto x:tree)cout<<x<<" ";cout<<endl;
        cout<<"##########################"<<endl;
    }
};

struct SegTree{
    int n;
    vector<int>tree;
    
    void build(int n_,int *arr){
        //n=(n_&(n_-1)?1<<(__lg(n_)+1):n_);
        if(__builtin_popcount(n_)!=1)
            n=1<<(__lg(n_)+1);
        else 
            n=n_;
        tree.resize(n<<1,0);
        //cout<<n<<" || "<<tree.size()<<endl;
        for(int i=0;i<n_;i++)
            tree[i+n]=arr[i];
        for(int i=n-1;i>=0;i--)
            tree[i]=merge(tree[i<<1],tree[i<<1 | 1]);
    }
    int merge(int l,int r){
        return l+r;
    }
    int query(int ql,int qr,int k,int sl,int sr){
        if(ql<=sl&&sr<=qr)
            return tree[k];
        if(qr<sl||sr<ql)
            return 0;
        int mid=(sl+sr)/2;
        return merge(query(ql,qr,k<<1,sl,mid),query(ql,qr,k<<1|1,mid+1,sr));
    }
    //point update
    void update(int ql,int qr,int v,int k,int sl,int sr){
        if(ql<=sl&&sr<=qr)
            return (void)(tree[k]=v);
        if(qr<sl||sr<ql)
            return;
        int mid=(sl+sr)/2;
        update(ql,qr,v,k<<1,sl,mid);
        update(ql,qr,v,k<<1|1,mid+1,sr);
        tree[k]=merge(tree[k<<1],tree[k<<1|1]);
    }
    void show(){
        /*for(auto x:tree)cout<<x.first<<" ";cout<<endl;
        for(auto x:tree)cout<<x.second<<" ";cout<<endl;
        cout<<"##########################"<<endl;*/
    }
};
struct segment_tree {
#define MID (start+end>>1)
	segment_tree *left, *right;
	int len = 0;
	ll sum = 0;
	int lazy_swap = 0;
	segment_tree(int len) :
			len(len) {
		if (len) {
			left = new segment_tree(len - 1);
			right = new segment_tree(len - 1);
		}
	}
	void push_down() {
		if (lazy_swap & (1 << len - 1)) {
			lazy_swap ^= (1 << len - 1);
			swap(left, right);
		}
		left->lazy_swap ^= lazy_swap;
		right->lazy_swap ^= lazy_swap;
		lazy_swap = 0;
	}
	ll query(int start, int end, int from, int to) {
		if (to < start || end < from)
			return 0;
		if (from <= start && end <= to)
			return sum;
		push_down();
		return left->query(start, MID, from, to)
				+ right->query(MID + 1, end, from, to);
	}
	void update(int start, int end, int pos, int val) {
		if (start == end) {
			sum = val;
			return;
		}
		push_down();
		if (pos <= MID)
			left->update(start, MID, pos, val);
		else
			right->update(MID + 1, end, pos, val);
		sum = left->sum + right->sum;
	}
};
// CHT opt..Lichao
struct Line {
	ll k, m;
	mutable ll p;
	bool operator<(const Line& o) const {
		return k < o.k;
	}
	bool operator<(const ll&x) const {
		return p < x;
	}
};
 
struct LineContainer : multiset<Line, less<>> {
	// (for doubles, use inf = 1/.0, div(a,b) = a/b)
	const ll inf = numeric_limits<ll>::max();
	ll div(ll a, ll b) { // floored division
		return a / b - ((a ^ b) < 0 && a % b);
	}
	bool isect(iterator x, iterator y) {
		if (y == end()) { x->p = inf; return false; }
		if (x->k == y->k) x->p = x->m > y->m ? inf : -inf;
		else x->p = div(y->m - x->m, x->k - y->k);
		return x->p >= y->p;
	}
	void add(ll k, ll m) {
		auto z = insert({k, m, 0}), y = z++, x = y;
		while (isect(y, z)) z = erase(z);
		if (x != begin() && isect(--x, y)) isect(x, y = erase(y));
		while ((y = x) != begin() && (--x)->p >= y->p)
			isect(x, erase(y));
	}
	ll query(ll x) {
		assert(!empty());
		auto l = *lower_bound(x);
		return l.k * x + l.m;
	}
};
//Sliding hash
struct Hash{
    ll pp1,m1,pp2,m2;
    int n,hash1=0,hash2=0;
    int HP1,HP2;
    Hash(ll PP1,ll M1,ll PP2,ll M2,string s){
        pp1=PP1;pp2=PP2;m1=M1;m2=M2;
        n=s.length();
        //init highest power..
        HP1=binpow(pp1,n-1,m1);
        HP2=binpow(pp2,n-1,m2);
        //build hash1
        for(int i=0;i<n;i++){
            hash1 = (mult(hash1,pp1,m1)+s[i])%m1;
            hash2 = (mult(hash2,pp2,m2)+s[i])%m2;
        }
    }
    void slide_next(char old,char nxt){
        hash1 -= mult(old,HP1,m1);
        if(hash1<0)hash1+=m1;
        hash1 = add(mult(hash1,pp1,m1),nxt,m1);

        hash2 -= mult(old,HP2,m2);
        if(hash2<0)hash2+=m2;
        hash2 = add(mult(hash2,pp2,m2),nxt,m2);
    }
    pair<int,int>get_hash(){
        return {hash1,hash2};
    }
    bool operator==(Hash &e)const{
        return hash1==e.hash1&&hash2==e.hash2;
    }
};
//prefix hash
const int N = 1550 + 3;
 
int pw1[N],pw2[N],inv1[N],inv2[N];
 
void Hash(int base1 = 59, int base2 = 69){
 
    pw1[0] = inv1[0]=1;
    int mininv1 = binpow(base1,mod-2,mod);
    for (int i =1; i<N; ++i){
        pw1[i] = mult(pw1[i-1],base1,mod);
        inv1[i] = mult(inv1[i-1],mininv1,mod);
    }
 
    pw2[0] = inv2[0]=1;
    int mininv2 = binpow(base2,mod-2,mod);
    for (int i =1; i<N; ++i){
        pw2[i] = mult(pw2[i-1],base2,mod);
        inv2[i] = mult(inv2[i-1],mininv2,mod);
    }
}
 
 
struct Hashing{
 
    vector <int>pre1,pre2;
    string s;
 
    Hashing(string &s){
        this->s = s;
        pre1 = pre2 = vector <int>(s.size());
        clc_pre_hash();
    }
 
    void clc_pre_hash(){
 
        int hv1 = 0,hv2=0;
        for (int i =0; i < s.size(); ++i){
            int c = s[i]-'a'+1;
            hv1 = add(hv1,mult(pw1[i],c,mod),mod);
            hv2 = add(hv2,mult(pw2[i],c,mod),mod);
            pre1[i] = hv1,pre2[i]=hv2;
        }
    }
 
    pair<int,int>get_hash_range(int L, int R){
        if (!L)
            return {pre1[R],pre2[R]};
        else{
            return {(mult(inv1[L],add(pre1[R],-pre1[L-1],mod),mod)+mod)%mod,
                    (mult(inv2[L],add(pre2[R],-pre2[L-1],mod),mod)+mod)%mod};
        }
    }
};
//KMP
struct KMP {
    string pattern;
    vector<int> prefix;

    KMP(const string& pattern) {
        this->pattern = pattern;
        buildPrefix();
    }

    void buildPrefix() {
        int n = pattern.length();
        prefix.resize(n);
        prefix[0] = 0;

        int j = 0;
        for (int i = 1; i < n; ++i) {
            if (pattern[i] == pattern[j]) {
                prefix[i] = j + 1;
                ++j;
            } else {
                if (j != 0) {
                    j = prefix[j - 1];
                    --i;  // Retry the current character
                } else {
                    prefix[i] = 0;
                }
            }
        }
    }

    int search(const string& text) {
        //vector<int> occurrences;
        int m = pattern.length();
        int n = text.length();
        int i = 0, j = 0;
        int cnt=0;
        while (i < n) {
            if (pattern[j] == text[i]) {
                ++i;
                ++j;
            }

            if (j == m) {
                //occurrences.push_back(i - j);
                cnt++;
                j = prefix[j - 1];
            } else if (i < n && pattern[j] != text[i]) {
                if (j != 0) {
                    j = prefix[j - 1];
                } else {
                    ++i;
                }
            }
        }

        return cnt;
    }
};
//Trie
class TrieNode {
public:
    unordered_map<char, TrieNode*> children;
    bool isEndOfWord;

    TrieNode() {
        isEndOfWord = false;
    }

    ~TrieNode() {
        for (auto& pair : children) {
            delete pair.second;
        }
    }
};

class Trie {
private:
    TrieNode* root;

public:
    Trie() {
        root = new TrieNode();
    }

    ~Trie() {
        delete root;
    }

    void insert(const string& word) {
        TrieNode* current = root;

        for (char ch : word) {
            if (current->children.find(ch) == current->children.end()) {
                current->children[ch] = new TrieNode();
            }
            current = current->children[ch];
        }

        current->isEndOfWord = true;
    }

    bool search(const string& word) {
        TrieNode* current = root;

        for (char ch : word) {
            if (current->children.find(ch) == current->children.end()) {
                return false;
            }
            current = current->children[ch];
        }

        return current->isEndOfWord;
    }

    bool startsWith(const string& prefix) {
        TrieNode* current = root;

        for (char ch : prefix) {
            if (current->children.find(ch) == current->children.end()) {
                return false;
            }
            current = current->children[ch];
        }

        return true;
    }
};
struct node{
    int t[2];
    int cnt;
    node(){cnt = 0 , memset(t , -1 , sizeof(t));}
};
struct bitrie {
  const ll LOG = 30;
  vector<node> trie;
  
  bitrie(){
    trie.clear();
    trie.resize(1);
  }
  int new_node(){
      trie.push_back(node());
      return trie.size() - 1;
  }
  void clear(){
    trie.clear();
    trie.resize(1);
  }
  void add(int val){
      int x = 0;
      for(int i = LOG; i >= 0; i--){
          trie[x].cnt++;
          int b = (bool)((1<<i) & val);
          if(trie[x].t[b] == -1) trie[x].t[b] = new_node();
          x = trie[x].t[b];
      }
      trie[x].cnt++;
  }
  void rem(int val){
      int x = 0;
      for(int i = LOG; i >= 0; i--){
          trie[x].cnt--;
          int b = (bool)((1ll<<i) & val);
          if(trie[x].t[b] == -1) trie[x].t[b] = new_node();
          x = trie[x].t[b];
      }
      trie[x].cnt--;
  }
  int max_xor(int val){
      int x = 0;
      int res = 0;
      for(int i = LOG; i >= 0; i--){
          if(x == -1) return 0;
          int b = ((val>>i)&1) ^ 1;
          if(trie[x].t[b] == -1 || trie[trie[x].t[b]].cnt == 0){
              x = trie[x].t[b^1];
          } else{
              x = trie[x].t[b];
              res |= (1ll<<i);
          }
      }
      return res;
  }
  int min_xor(int val){
    int x = 0;
    int res = 0;
    for(int i = LOG; i >= 0; i--){
        if(x == -1) return ll(2e9) + 5;
        int b = ((val>>i)&1);
        if(trie[x].t[b] != -1 and trie[trie[x].t[b]].cnt != 0){
          x = trie[x].t[b];
        } else {
          if(trie[x].t[b ^ 1] != -1)
            res |= (1ll << i);
          x = trie[x].t[b ^ 1];
        }
    }
    return res;
  }
};
 
//Trie bitmask
class TrieNode {
public:
    TrieNode* children[2];
    bool isEndOfWord;

    TrieNode() {
        children[0]=children[1]=0;
        isEndOfWord = false;
    }

    ~TrieNode() {
        delete children[0];
        delete children[1];
    }
};

class Trie {
private:
    TrieNode* root;

public:
    Trie() {
        root = new TrieNode();
    }

    ~Trie() {
        delete root;
    }

    void insert(const int& mask) {
        TrieNode* current = root;

        for (int d=30;d>=0;d--) {
            int ch=!!(mask&(1<<d));
            if (current->children[ch]==0) {
                current->children[ch] = new TrieNode();
            }
            current = current->children[ch];
        }

        current->isEndOfWord = true;
    }
    /*
        search for max number which gives max mask.. 
    */
    int search(const int& mask) {
        TrieNode* current = root;
        int ans=mask;
        for (int d=30;d>=0;d--) {
            int ch=!(mask&(1<<d));
            if (current->children[ch]) {
                ans ^=(ch<<d);
                current = current->children[ch];    
            }
            else if(current->children[!ch]){
                ans ^=((!ch)<<d);
                current = current->children[!ch];
            }
            else{
                return ans;
            }
            
        }

        return ans;
    }
    void erase(const int& mask){
        TrieNode* current = root,*prev;
        for (int d=30;d>=0;d--) {
            int ch=!!(mask&(1<<d));
            if (current->children[ch]==0) {
                return;//not found?! not possible..
            }
            prev=current;
            current = current->children[ch];
            current->cnt--;
            if(current->cnt==0){
                //we must delete this pointer..
                prev->children[ch]=0;
                //using memory deallocation probess ,may work?! 
                delete current;
                break;
            }
        }
        
        
    }
};
//LCA
const int LG = 18;
int depth[OL];
int SUC[LG][OL];
int ACU[OL];
void dfs_LCA(int u , int p){
    for(auto e : adj[u]){
        int nxt = e.first , eid = e.second;
        if(nxt == p)continue;
        depth[nxt] = depth[u] + 1;
        SUC[0][nxt] = u;
        for(int d = 1;d < LG;d++){
            SUC[d][nxt] = SUC[d - 1][SUC[d - 1][nxt]];
        }
        dfs_LCA(nxt , u);
    }
}
int get_kth(int x, int k){
    
    for(int d = 0;d < LG;d++){
        if(k & (1<<d))x = SUC[d][x];
    }
    if(x < 1)return 1;
    return x;
}
int get_LCA(int u ,int v){
    if(depth[u] < depth[v])swap(u , v);
    u = get_kth(u , depth[u] - depth[v]);
    if(u == v)return u;
    for(int d = LG - 1;~d;d--){
        if(SUC[d][u] != SUC[d][v]){
            u = SUC[d][u];v = SUC[d][v];
        }        
    }

    return SUC[0][u];
}

//EGCD
void next_r(ll &r0 , ll &r1 , ll r){
    ll r2 = r0 - r1 * r;
    r0 = r1;
    r1 = r2;
}
ll egcd(ll r0 , ll r1 , ll &x0 , ll &y0){
    ll x1 = y0 = 0 , y1 = x0 = 1;
    while(r1){
        ll r = r0 / r1;
        next_r(r0 , r1 , r);
        next_r(x0 , x1 , r);
        next_r(y0 , y1 , r);
    }
    return r0;
}
// C = X * A + Y * B
// X' = X - (B / G) * K , Y' = Y + (A / G) * K  :> For any K 
ll solveLDE(ll a , ll b , ll c , ll &x , ll &y , ll &g){ 
    g = egcd(a , b , x , y);
    ll m = c / g;
    x *= m; y *= m;
    return m * g == c;
}
struct mod_eq{
    ll r , m;
    mod_eq(ll rem , ll mod){
        r = rem;
        m = mod;
    }
    mod_eq(){}

};
mod_eq CRT(const mod_eq &e1 , const mod_eq &e2){
    ll q1 , q2 , g;
    if(!solveLDE(e1.m , -e2.m , e2.r - e1.r ,q1 , q2 , g)){
        throw "No Solution";
    }
    q2 %= e1.m / g;
    ll lcm = abs(e1.m / g * e2.m);
    ll x = e2.m * q2 + e2.r;
    x %= lcm;
    if(x < 0)x += lcm;
    return mod_eq(x , lcm);
}
//PHI
iota(PHI , PHI + OL , 0);
for(int i = 2;i < OL;i++){
    if(i != PHI[i])continue;
    for(int j = i;j < OL;j += i){
        PHI[j] -= PHI[j] / i;
    }
}
int primes[664580],cnt;
bool vis[OL];
void p_seive(){
    cnt = 0;
    for(ll i = 2;i < OL;i++){
        if(vis[i])continue;
        primes[cnt++] = i;
        for(ll j = i * i;j < OL;j += i)vis[j] = 1;
    }
}
//segmented phi sieve ..!!!
void pre_phi(ll a ,ll b){
    int n = b - a + 1;
    ll phi[n + 1];
    ll large_num[n + 1];
    iota(phi , phi + n + 1 , a);
    iota(large_num , large_num + n + 1 , a);
    for(ll i = 0;i < cnt && primes[i] <= b / primes[i];i++){
        for(ll j = ((a + primes[i] - 1) / primes[i]) * primes[i] ; j <= b;j += primes[i]){
            phi[j - a] -= phi[j - a] / primes[i];
            do{
                large_num[j - a] /= primes[i];
            }while(!(large_num[j - a] % primes[i]));
        }
    }
    for(int i = 0;i < n;i++){
        if(large_num[i] > 1){
            phi[i] -= phi[i] / large_num[i];
        }
        cout<<phi[i]<<endl;
    }
}
//matrix expo
vector <vector<int>>mulMat(vector <vector <int>>&a, vector <vector <int>>&b){
    int r1 = a.size(), c1 = a[0].size();
    int r2 = b.size(), c2 = b[0].size();

    vector <vector <int>>res(r1,vector <int>(c2));
    for (int i = 0; i < r1; ++i){
        for (int j = 0; j < c2; ++j){
            for (int k = 0; k < c1; ++k){
                res[i][j] = add(res[i][j],mult(a[i][k],b[k][j],mod),mod);
                //res[i][j] = res[i][j] + a[i][k] * b[k][j];
            }
        }
    }

    return res;
}

vector <vector <int>>FPMatrix(vector <vector <int>>&a,ll n){
    int k = a.size();
    vector <vector <int>>res(k,vector <int>(k));

    for (int i = 0; i< k; ++i)
        res[i][i]=1;

    while(n){
        if (n&1)res= mulMat(res,a);
        a = mulMat(a,a);
        n>>=1;
    }
    return res;
}

//Lazy seg
struct Node{
    int mx = 0;
    int lazy = 0;
    Node(){}
};
struct SegTree{
    int n;
    vector<Node>tree;
    
    void build(int n_,int *arr){
        if(__builtin_popcount(n_)!=1)
            n=1<<(__lg(n_)+1);
        else 
            n=n_;
        tree.resize(n<<1,Node());
        for(int i=n-1;i>=0;i--)
            tree[i]=merge(tree[i<<1],tree[i<<1 | 1]);
    }
    void push(int node, int l, int r) {
        tree[node].mx += tree[node].lazy;
        if (l < r) { // not a leaf node
            tree[node << 1].lazy     += tree[node].lazy;
            tree[node << 1 | 1].lazy += tree[node].lazy;
        }
        tree[node].lazy = 0;
    }
    Node merge(Node l,Node r){
        Node res;
        res.mx = max(l.mx , r.mx);
        return res;
    }
    Node query(int ql,int qr,int k,int sl,int sr){
        push(k , sl , sr);
        if(ql<=sl&&sr<=qr)
            return tree[k];
        if(qr<sl||sr<ql)
            return Node();
        int mid=(sl+sr)/2;
        return merge(query(ql,qr,k<<1,sl,mid),query(ql,qr,k<<1|1,mid+1,sr));
    }

    void update(int ql,int qr,int v,int k,int sl,int sr){
        push(k , sl , sr);
        if(ql<=sl&&sr<=qr)
            {
                tree[k].lazy += v;
                push(k , sl , sr);
                return;
            }
        if(qr<sl||sr<ql)
            return;
        int mid=(sl+sr)/2;
        update(ql,qr,v,k<<1,sl,mid);
        update(ql,qr,v,k<<1|1,mid+1,sr);
        tree[k]=merge(tree[k<<1],tree[k<<1|1]);
    }
    
};
const int MAX_N = 3e5 + 3;
ll Tree[MAX_N * 4];
ll lazy[MAX_N * 4];
vector<ll> v;

void build(int node, int l, int r) {
    if (l == r) {
        Tree[node] = v[l];
        return;
    }
    int mid = (l + r) / 2;
    build(node * 2, l, mid);
    build(node * 2 + 1, mid + 1, r);
    Tree[node] = min(Tree[node * 2], Tree[node * 2 + 1]);
}

void push(int node, int l, int r) {
    Tree[node] += lazy[node];
    if (l < r) { // not a leaf node
        lazy[node * 2]     += lazy[node];
        lazy[node * 2 + 1] += lazy[node];
    }
    lazy[node] = 0;
}

void update(int node, int l, int r, int ql, int qr, ll nval) {
    push(node, l, r);
    if (ql > r || qr < l)
        return;
    if (ql <= l && r <= qr) {
        lazy[node] += nval;
        push(node, l, r);
        return;
    }
    int mid = (l + r) / 2;
    update(node * 2, l, mid, ql, qr, nval);
    update(node * 2 + 1, mid + 1, r, ql, qr, nval);
    Tree[node] = min(Tree[node * 2], Tree[node * 2 + 1]);
}
ll query(int node, int l, int r, int ql, int qr) {
    // First, push down any lazy updates for the current node.
    push(node, l, r);
    // No overlap.
    if (ql > r || qr < l) return LLONG_MAX; // Use a very large value.
    // Total overlap.
    if (ql <= l && r <= qr) return Tree[node];
    // Partial overlap.
    int mid = (l + r) / 2;
    return min(query(node * 2, l, mid, ql, qr),
               query(node * 2 + 1, mid + 1, r, ql, qr));
}
void buildSegmentTree(const vector<ll>& arr) {
    v = arr;  // assign input array to global vector v
    int n = v.size();
    fill(Tree, Tree + n * 4, 1000);
    fill(lazy, lazy + n * 4, 0);
    build(1, 0, n - 1);
}


int k;
//query range - element
ll query(int node, int l , int r, int ql, int qr){

    push(node,l,r);

    if (qr < l || ql > r )
        return 1e15+7;

    if (l>=ql && r<=qr) {
        return Tree[node];
    }
    int mid = (l+r)/2;


    return min(query(node<<1,l, mid,ql,qr) , query(node<<1|1, mid+1,r,ql, qr));
}
//SPARSE TABLE RMQ
int n, LOG = 30;
vector<vector<int>> st; // Sparse table

// Build the sparse table for range maximum queries.
void buildSparseTable(int arr[], int n) {
    LOG = floor(log2(n)) + 1;
    st.assign(n, vector<int>(LOG));
    
    // Initialize the 0th level with the original array values.
    for (int i = 0; i < n; i++) {
        st[i][0] = arr[i];
    }
    
    // Build the table: st[i][j] holds the max of the interval starting at i of length 2^j.
    for (int j = 1; j < LOG; j++) {
        for (int i = 0; i + (1 << j) <= n; i++) {
            st[i][j] = max(st[i][j-1], st[i + (1 << (j-1))][j-1]);
        }
    }
}

// Query the maximum value in the range [L, R] (0-indexed).
int queryMax(int L, int R) {
    int j = floor(log2(R - L + 1));
    return max(st[L][j], st[R - (1 << j) + 1][j]);
}
// F(C) = sum{a xor b = c} ( G(a) * D(b)) ===> convert them to walsh domain , convolution becomes multiplication , then do inverse FHWT ez.. 
// Fast Walsh-Hadamard Transform for XOR convolution.
void fwht(vector<ll>& a, bool invert) {
    int n = a.size();
    for (int len = 1; 2 * len <= n; len <<= 1) {
        for (int i = 0; i < n; i += 2 * len) {
            for (int j = 0; j < len; j++) {
                ll u = a[i + j], v = a[i + j + len];
                a[i + j] = u + v;
                a[i + j + len] = u - v;
            }
        }
    }
    if (invert) {
        for (int i = 0; i < n; i++) {
            a[i] /= n;
        }
    }
}
 int mex(std::vector<int>& a) {
    int n = a.size();
    for (int i = 0; i < n; ++i) {
        // Place a[i] to its correct position if in [0, n)
        while (a[i] >= 0 && a[i] < n && a[i] != a[a[i]]) {
            std::swap(a[i], a[a[i]]);
        }
    }
    for (int i = 0; i < n; ++i) {
        if (a[i] != i)
            return i;
    }
    return n;
}
// enhanced ordered set ordered multiset
#include <ext/pb_ds/tree_policy.hpp>
#include <ext/pb_ds/assoc_container.hpp>
 
using namespace std;
namespace __gnu_pbds{
          typedef tree<ll,
                       null_type,
                       less_equal<ll>,
                       rb_tree_tag,
                       tree_order_statistics_node_update> ordered_set;
}
using namespace __gnu_pbds;
 
void Insert(ordered_set &s,ll x){ //this function inserts one more occurrence of (x) into the set.
     s.insert(x);
}
 
 
bool Exist(ordered_set &s,ll x){ //this function checks weather the value (x) exists in the set or not.
     if((s.upper_bound(x))==s.end()){
        return 0;
     }
     return ((*s.upper_bound(x))==x);
}
 
 
void Erase(ordered_set &s,ll x){ //this function erases one occurrence of the value (x).
     if(Exist(s,x)){
        s.erase(s.upper_bound(x));
     }
}
 
 
ll FirstIdx(ordered_set &s,ll x){ //this function returns the first index of the value (x)..(0 indexing).
    if(!Exist(s,x)){
        return -1;
    }
    return (s.order_of_key(x));
}
 
 
ll Value(ordered_set &s,ll idx){ //this function returns the value at the index (idx)..(0 indexing).
   return (*s.find_by_order(idx));
}
 
 
ll LastIdx(ordered_set &s,ll x){ //this function returns the last index of the value (x)..(0 indexing).
    if(!Exist(s,x)){
        return -1;
    }
    if(Value(s,(int)s.size()-1)==x){
        return (int)(s.size())-1;
    }
    return FirstIdx(s,*s.lower_bound(x))-1;
}
ll Count(ordered_set &s,ll x){ //this function returns the number of occurrences of the value (x).
     if(!Exist(s,x)){
        return 0;
     }
     return LastIdx(s,x)-FirstIdx(s,x)+1;
}
void Clear(ordered_set &s){ //this function clears all the elements from the set.
     s.clear();
}
 
ll Size(ordered_set &s){ //this function returns the size of the set.
     return (int)(s.size());
}
ll how_many_smaller_equal(ordered_set &s , ll x){
    auto it = s.lower_bound(x);
    if(it == s.end()){
        return s.size();
    }
    ll idx = FirstIdx(s , *it);
    return idx;
}
//MAX FLOW
// Edmonds karp O(V E^2)
int n , m;
ll cap[501][501];
vector<int>adj[501];
int anc[501];
ll get_flow(int s , int t){
    if(s == t)return 0;
    auto reach = [&]()->bool
    {
        memset(anc , -1 , sizeof(int) * (n + 1));
        deque<int> qu;
        qu.push_back(s);
        
        while(!qu.empty()){
            int node = qu.front();
            qu.pop_front();
            if(node == t)break;
            for(auto nxt : adj[node]){
                if(cap[node][nxt] < 1 || anc[nxt] != -1 || nxt == s)continue;
                anc[nxt] = node;
                qu.push_back(nxt);
                
            }
        }
        return anc[t] != -1;
    };
    ll flow = 0;
    while(reach()){
        int cur = t;
        ll mn = INF;
        while(cur != s){
            int par = anc[cur];
            chmin(mn , cap[par][cur]);
            cur = par;
        }
        cur = t;
        while(cur != s){
            int par = anc[cur];
            cap[par][cur] -= mn;
            cap[cur][par] += mn;
            cur = par;
        }
        flow += mn;
    }
    return flow;
}
// Maximum Bipartite matching 
class HopcroftKarp {
    int U, V;
    vector<vector<int>> adj;
    vector<int> pairU, pairV, dist;

public:
    HopcroftKarp(int U, int V) : U(U), V(V), adj(U + 1), pairU(U + 1, 0), pairV(V + 1, 0), dist(U + 1) {}

    void addEdge(int u, int v) {
        adj[u].push_back(v);
    }

    bool bfs() {
        queue<int> q;
        for (int u = 1; u <= U; ++u) {
            if (pairU[u] == 0) {
                dist[u] = 0;
                q.push(u);
            } else {
                dist[u] = INT_MAX;
            }
        }
        dist[0] = INT_MAX;
        while (!q.empty()) {
            int u = q.front(); q.pop();
            if (dist[u] < dist[0]) {
                for (int v : adj[u]) {
                    if (dist[pairV[v]] == INT_MAX) {
                        dist[pairV[v]] = dist[u] + 1;
                        q.push(pairV[v]);
                    }
                }
            }
        }
        return dist[0] != INT_MAX;
    }

    bool dfs(int u) {
        if (u == 0) return true;
        for (int v : adj[u]) {
            if (dist[pairV[v]] == dist[u] + 1 && dfs(pairV[v])) {
                pairU[u] = v;
                pairV[v] = u;
                return true;
            }
        }
        dist[u] = INT_MAX;
        return false;
    }

    int maxMatching() {
        int matching = 0;
        while (bfs()) {
            for (int u = 1; u <= U; ++u) {
                if (pairU[u] == 0 && dfs(u)) {
                    matching++;
                    
                }
            }
        }
        
        return matching;
    }
};