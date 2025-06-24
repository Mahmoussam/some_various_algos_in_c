  #ifdef ONLINE_JUDGE
        freopen("input.txt", "r", stdin);
        freopen("output.txt", "w", stdout);
  #endif
// 0x3f3f3f3f
#include<bits/stdc++.h>

#pragma GCC optimize("O3", "unroll-loops", "inline-functions")
//FFT optimizatiion flags:
#pragma GCC target("avx2", "bmi2", "fma", "popcnt", "sse4.2")

using namespace std;
#define all(X) X.begin(),X.end()
#define rall(X) X.rbegin(),X.rend()
 
#define ll long long
 
ll mod =  998244353;
 
ll mult(ll a, ll b) { return (a % mod * b % mod) % mod; }
ll add(ll a, ll b) { return (a % mod + b % mod) % mod; }
ll binpow_m(ll base,ll exp){ll res=1;while(exp>0){if(exp & 1)res=mult(res,base);base= mult(base,base);exp>>=1;}return res;}
ll binpow(ll base,ll exp){ll res=1;while(exp>0){if(exp & 1)res=res*base;base= base*base;exp>>=1;}return res;}
ll lcm(ll a , ll b){return a * b / __gcd(a , b);}
 
template <typename T> static inline void chmax(T &a , T b){a = max(a , b);}
template <typename T> static inline void chmin(T &a , T b){a = min(a , b);}
 //fast input
    ios_base::sync_with_stdio(false);
    cin.tie(0);
    cout.tie(0);

//random  xor hashing
const ll SEED = chrono::steady_clock::now().time_since_epoch().count();
mt19937_64 rng(SEED);
inline ll rnd(ll l = 0, ll r = INF)
{
  return uniform_int_distribution<ll>(l, r)(rng);
}
struct Hash {
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
vector<int> vec = {1 , 2 , 3 , 4 ,  4, 4 , 6 , 10};
cout << (int)(lower_bound(vec.begin(), vec.end() , 4) - vec.begin()) << endl;

#include <bits/stdc++.h>

using namespace std;
#define ll long long
int Nums[5];
void initiaition();
void prime();
struct st {
  int x, y;
  st(int xx, int yy) {
    x = xx;
    y = yy;
  }
  st() {}
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
// Fenwick BIT
ll BIT[11] = {0};
ll n[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
ll query(int i) {
  ll sum = 0;
  while (i > 0) {
    sum += BIT[i];
    i -= (i & -i); //remove last bit
  }
  return sum;
}
void update(int i, ll k) {
  while (i < 11) {
    BIT[i] += k;
    i += (i & -i);
  }
}
//BIT INVERSION
ll query(int i, vector<int>&BIT) {
  ll sum = 0;
  while (i < BIT.size()) {
    sum += BIT[i];
    i += (i & -i); //remove last bit
  }
  return sum;
}
void update(int i, ll k , vector<int>&BIT) {
  while (i > 0) {
    BIT[i] += k;
    i -= (i & -i);
  }
}
void prime() {

  for (int i = 0; i < 5 ; i++) {
    bool isPrime = true;
    for (int j = 2; j < Nums[i] ; j++) {
      if (Nums[i] % j == 0)
        isPrime = false;

    }
    if (Nums[i] == 0 || Nums[i] == 1)
      isPrime = false;
    if (isPrime == true)
      cout << Nums[i] << " is a prime number." << endl;
  }
}
//CHT
//x:start pos ,m,b==>y=m*x+b
struct seg {
  long double x;
  ll m, b;
};
vector<seg>hull;
ll query(ll x) {
  if (!hull.size())return 0;
  seg s = *--upper_bound(hull.begin(), hull.end(), x, [](ll a, seg b) {
    return a < b.x;
  });
  return s.b + s.m * x;
}
void insert(ll m, ll b) {
  while (hull.size()) {
    seg s = hull.back();
    if (s.b + s.m * s.x > b + m * s.x) {
      if (s.m - m)hull.push_back({(b - s.b) / (long double)(s.m - m), m, b});
      return;
    }
    hull.pop_back();
  }
  hull = {{LLONG_MIN, m, b}};
}
struct SegTree {
  vector<int>tree;
  int n;
  void build(int oldn, int *arr) {
    if (__builtin_popcount(oldn) != 1)
      this->n = 1 << (__lg(oldn) + 1);
    else
      this->n = oldn;
    tree.resize(this->n << 1, 1e9);
    for (int i = 0; i < oldn; i++)
      tree[i + this->n] = arr[i];
    for (int i = this->n - 1; i >= 0; i--) {
      tree[i] = min(tree[i << 1], tree[i << 1 | 1]);
    }
  }
  int query(int ql, int qr, int k, int sl, int sr) {
    //if seg inside ,return seg
    if (sl >= ql && sr <= qr)return tree[k];
    if (ql > sr || qr < sl)return 1e9;
    int mid = (sl + sr) / 2;
    return min(query(ql, qr, k << 1, sl, mid), query(ql, qr, k << 1 | 1 , mid + 1, sr));
  }
  //point update
  void update(int ql, int qr, int v, int k, int sl, int sr) {
    //if seg inside ,return seg
    if (sl >= ql && sr <= qr) {
      tree[k] = v;
      return;
    }
    if (ql > sr || qr < sl)return;
    int mid = (sl + sr) / 2;
    update(ql, qr, v, k << 1, sl, mid);
    update(ql, qr, v, k << 1 | 1 , mid + 1, sr);
    //re calculate
    tree[k] = min(tree[k << 1], tree[k << 1 | 1]);
  }
  void show() {
    for (auto x : tree)cout << x << " "; cout << endl;
    cout << "##########################" << endl;
  }
};

struct SegTree {
  int n;
  vector<int>tree;

  void build(int n_, int *arr) {
    //n=(n_&(n_-1)?1<<(__lg(n_)+1):n_);
    if (__builtin_popcount(n_) != 1)
      n = 1 << (__lg(n_) + 1);
    else
      n = n_;
    tree.resize(n << 1, 0);
    //cout<<n<<" || "<<tree.size()<<endl;
    for (int i = 0; i < n_; i++)
      tree[i + n] = arr[i];
    for (int i = n - 1; i >= 0; i--)
      tree[i] = merge(tree[i << 1], tree[i << 1 | 1]);
  }
  int merge(int l, int r) {
    return l + r;
  }
  int query(int ql, int qr, int k, int sl, int sr) {
    if (ql <= sl && sr <= qr)
      return tree[k];
    if (qr < sl || sr < ql)
      return 0;
    int mid = (sl + sr) / 2;
    return merge(query(ql, qr, k << 1, sl, mid), query(ql, qr, k << 1 | 1, mid + 1, sr));
  }
  //point update
  void update(int ql, int qr, int v, int k, int sl, int sr) {
    if (ql <= sl && sr <= qr)
      return (void)(tree[k] = v);
    if (qr < sl || sr < ql)
      return;
    int mid = (sl + sr) / 2;
    update(ql, qr, v, k << 1, sl, mid);
    update(ql, qr, v, k << 1 | 1, mid + 1, sr);
    tree[k] = merge(tree[k << 1], tree[k << 1 | 1]);
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
    if (y == end()) {
      x->p = inf;
      return false;
    }
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
struct Hash {
  ll pp1, m1, pp2, m2;
  int n, hash1 = 0, hash2 = 0;
  int HP1, HP2;
  Hash(ll PP1, ll M1, ll PP2, ll M2, string s) {
    pp1 = PP1; pp2 = PP2; m1 = M1; m2 = M2;
    n = s.length();
    //init highest power..
    HP1 = binpow(pp1, n - 1, m1);
    HP2 = binpow(pp2, n - 1, m2);
    //build hash1
    for (int i = 0; i < n; i++) {
      hash1 = (mult(hash1, pp1, m1) + s[i]) % m1;
      hash2 = (mult(hash2, pp2, m2) + s[i]) % m2;
    }
  }
  void slide_next(char old, char nxt) {
    hash1 -= mult(old, HP1, m1);
    if (hash1 < 0)hash1 += m1;
    hash1 = add(mult(hash1, pp1, m1), nxt, m1);

    hash2 -= mult(old, HP2, m2);
    if (hash2 < 0)hash2 += m2;
    hash2 = add(mult(hash2, pp2, m2), nxt, m2);
  }
  pair<int, int>get_hash() {
    return {hash1, hash2};
  }
  bool operator==(Hash &e)const {
    return hash1 == e.hash1 && hash2 == e.hash2;
  }
};
//prefix hash
const int N = 4005 + 3;
int pw1[N], pw2[N], inv1[N], inv2[N];
void Hash(int base1 = 59, int base2 = 69) {
  pw1[0] = inv1[0] = 1;
  int mininv1 = binpow_m(base1, mod - 2);
  for (int i = 1; i < N; ++i) {
    pw1[i] = mult(pw1[i - 1], base1);
    inv1[i] = mult(inv1[i - 1], mininv1);
  }

  pw2[0] = inv2[0] = 1;
  int mininv2 = binpow_m(base2, mod - 2);
  for (int i = 1; i < N; ++i) {
    pw2[i] = mult(pw2[i - 1], base2);
    inv2[i] = mult(inv2[i - 1], mininv2);
  }
}

struct Hashing {

  vector <int>pre1, pre2;
  string s;

  Hashing(string &s) {
    this->s = s;
    pre1 = pre2 = vector <int>(s.size());
    clc_pre_hash();
  }

  void clc_pre_hash() {

    int hv1 = 0, hv2 = 0;
    for (int i = 0; i < s.size(); ++i) {
      int c = s[i] - 'a' + 1;
      hv1 = add(hv1, mult(pw1[i], c));
      hv2 = add(hv2, mult(pw2[i], c));
      pre1[i] = hv1, pre2[i] = hv2;
    }
  }

  pair<int, int>get_hash_range(int L, int R) {
    if (!L)
      return {pre1[R], pre2[R]};
    else {
      return {(mult(inv1[L], add(pre1[R], -pre1[L - 1])) + mod) % mod,
              (mult(inv2[L], add(pre2[R], -pre2[L - 1])) + mod) % mod};
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
    int cnt = 0;
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

//Trie bitmask
class TrieNode {
  public:
    TrieNode* children[2];
    bool isEndOfWord;

    TrieNode() {
      children[0] = children[1] = 0;
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

      for (int d = 30; d >= 0; d--) {
        int ch = !!(mask & (1 << d));
        if (current->children[ch] == 0) {
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
      int ans = mask;
      for (int d = 30; d >= 0; d--) {
        int ch = !(mask & (1 << d));
        if (current->children[ch]) {
          ans ^= (ch << d);
          current = current->children[ch];
        }
        else if (current->children[!ch]) {
          ans ^= ((!ch) << d);
          current = current->children[!ch];
        }
        else {
          return ans;
        }
      }
      return ans;
    }
    void erase(const int& mask) {
      TrieNode* current = root, *prev;
      for (int d = 30; d >= 0; d--) {
        int ch = !!(mask & (1 << d));
        if (current->children[ch] == 0) {
          return;//not found?! not possible..
        }
        prev = current;
        current = current->children[ch];
        current->cnt--;
        if (current->cnt == 0) {
          //we must delete this pointer..
          prev->children[ch] = 0;
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
void dfs_LCA(int u , int p) {
  for (auto e : adj[u]) {
    int nxt = e.first , eid = e.second;
    if (nxt == p)continue;
    depth[nxt] = depth[u] + 1;
    SUC[0][nxt] = u;
    for (int d = 1; d < LG; d++) {
      SUC[d][nxt] = SUC[d - 1][SUC[d - 1][nxt]];
    }
    dfs_LCA(nxt , u);
  }
}
int get_kth(int x, int k) {
  for (int d = 0; d < LG; d++) {
    if (k & (1 << d))x = SUC[d][x];
  }
  if (x < 1)return 1;
  return x;
}
int get_LCA(int u , int v) {
  if (depth[u] < depth[v])swap(u , v);
  u = get_kth(u , depth[u] - depth[v]);
  if (u == v)return u;
  for (int d = LG - 1; ~d; d--) {
    if (SUC[d][u] != SUC[d][v]) {
      u = SUC[d][u]; v = SUC[d][v];
    }
  }
  return SUC[0][u];
}
int get_kth_on_path(int a, int b, int k)
{
	int lc = get_LCA(a, b);
	int d1 = depth[a] - depth[lc];
	int d2 = depth[b] - depth[lc];
	if(k <= d1){
		return get_kth(a , k);
	}else
		return get_kth(b , d2 + d1 - k);
}
//EGCD
void next_r(ll &r0 , ll &r1 , ll r) {
  ll r2 = r0 - r1 * r;
  r0 = r1;
  r1 = r2;
}
ll egcd(ll r0 , ll r1 , ll &x0 , ll &y0) {
  ll x1 = y0 = 0 , y1 = x0 = 1;
  while (r1) {
    ll r = r0 / r1;
    next_r(r0 , r1 , r);
    next_r(x0 , x1 , r);
    next_r(y0 , y1 , r);
  }
  return r0;
}
// C = X * A + Y * B
// X' = X - (B / G) * K , Y' = Y + (A / G) * K  :> For any K
ll solveLDE(ll a , ll b , ll c , ll &x , ll &y , ll &g) {
  g = egcd(a , b , x , y);
  ll m = c / g;
  x *= m; y *= m;
  return m * g == c;
}
struct mod_eq {
  ll r , m;
  mod_eq(ll rem , ll mod) {
    r = rem;
    m = mod;
  }
  mod_eq() {}
};
mod_eq CRT(const mod_eq &e1 , const mod_eq &e2) {
  ll q1 , q2 , g;
  if (!solveLDE(e1.m , -e2.m , e2.r - e1.r , q1 , q2 , g)) {
    throw "No Solution";
  }
  q2 %= e1.m / g;
  ll lcm = abs(e1.m / g * e2.m);
  ll x = e2.m * q2 + e2.r;
  x %= lcm;
  if (x < 0)x += lcm;
  return mod_eq(x , lcm);
}
//PHI
iota(PHI , PHI + OL , 0);
for (int i = 2; i < OL; i++) {
  if (i != PHI[i])continue;
  for (int j = i; j < OL; j += i) {
    PHI[j] -= PHI[j] / i;
  }
}
int primes[664580], cnt;
bool vis[OL];
void p_seive() {
  cnt = 0;
  for (ll i = 2; i < OL; i++) {
    if (vis[i])continue;
    primes[cnt++] = i;
    for (ll j = i * i; j < OL; j += i)vis[j] = 1;
  }
}
//segmented phi sieve ..!!!
void pre_phi(ll a , ll b) {
  int n = b - a + 1;
  ll phi[n + 1];
  ll large_num[n + 1];
  iota(phi , phi + n + 1 , a);
  iota(large_num , large_num + n + 1 , a);
  for (ll i = 0; i < cnt && primes[i] <= b / primes[i]; i++) {
    for (ll j = ((a + primes[i] - 1) / primes[i]) * primes[i] ; j <= b; j += primes[i]) {
      phi[j - a] -= phi[j - a] / primes[i];
      do {
        large_num[j - a] /= primes[i];
      } while (!(large_num[j - a] % primes[i]));
    }
  }
  for (int i = 0; i < n; i++) {
    if (large_num[i] > 1) {
      phi[i] -= phi[i] / large_num[i];
    }
    cout << phi[i] << endl;
  }
}
//matrix expo
template<typename T, size_t R1, size_t C1, size_t R2, size_t C2>
auto mulMat(const array<array<T, C1>, R1>& a, const array<array<T, C2>, R2>& b )
-> array<array<T, C2>, R1> {
  array<array<T, C2>, R1> res{};
  for (size_t i = 0; i < R1; ++i) {
    for (size_t j = 0; j < C2; ++j) {
      for (size_t k = 0; k < C1; ++k) {
        res[i][j] = add(res[i][j], mult(a[i][k], b[k][j] ) );
      }
    }
  }
  return res;
}
// Generic fast exponentiation for square matrices
template<typename T, size_t N>
std::array<std::array<T, N>, N> FPMatrix(std::array<std::array<T, N>, N> a, ll n) {
  std::array<std::array<T, N>, N> res{};
  // Initialize res as the identity matrix.
  for (size_t i = 0; i < N; ++i) {
    res[i][i] = 1;
  }
  while (n) {
    if (n & 1)
      res = mulMat(res, a);
    a = mulMat(a, a);
    n >>= 1;
  }
  return res;
}
vector <vector<int>>mulMat(vector <vector <int>>&a, vector <vector <int>>&b) {
  int r1 = a.size(), c1 = a[0].size();
  int r2 = b.size(), c2 = b[0].size();
  vector <vector <int>>res(r1, vector <int>(c2));
  for (int i = 0; i < r1; ++i) {
    for (int j = 0; j < c2; ++j) {
      ll sum = 0;
      for (int k = 0; k < c1; ++k) {
        (sum +=  ((ll)a[i][k])* (b[k][j]))%=mod;
        //res[i][j] = add(res[i][j], mult(a[i][k], b[k][j], mod), mod);
        //res[i][j] = res[i][j] + a[i][k] * b[k][j];
      }
      res[i][j] = sum;
    }
  }

  return res;
}

vector <vector <int>>FPMatrix(vector <vector <int>>&a, ll n) {
  int k = a.size();
  vector <vector <int>>res(k, vector <int>(k));

  for (int i = 0; i < k; ++i)
    res[i][i] = 1;

  while (n) {
    if (n & 1)res = mulMat(res, a);
    a = mulMat(a, a);
    n >>= 1;
  }
  return res;
}

//Lazy seg
struct Node {
  int mx = 0;
  int lazy = 0;
  Node() {}
};
struct SegTree {
  int n;
  vector<Node>tree;

  void build(int n_, int *arr) {
    if (__builtin_popcount(n_) != 1)
      n = 1 << (__lg(n_) + 1);
    else
      n = n_;
    tree.resize(n << 1, Node());
    for (int i = n - 1; i >= 0; i--)
      tree[i] = merge(tree[i << 1], tree[i << 1 | 1]);
  }
  void push(int node, int l, int r) {
    tree[node].mx += tree[node].lazy;
    if (l < r) { // not a leaf node
      tree[node << 1].lazy     += tree[node].lazy;
      tree[node << 1 | 1].lazy += tree[node].lazy;
    }
    tree[node].lazy = 0;
  }
  Node merge(Node l, Node r) {
    Node res;
    res.mx = max(l.mx , r.mx);
    return res;
  }
  Node query(int ql, int qr, int k, int sl, int sr) {
    push(k , sl , sr);
    if (ql <= sl && sr <= qr)
      return tree[k];
    if (qr < sl || sr < ql)
      return Node();
    int mid = (sl + sr) / 2;
    return merge(query(ql, qr, k << 1, sl, mid), query(ql, qr, k << 1 | 1, mid + 1, sr));
  }

  void update(int ql, int qr, int v, int k, int sl, int sr) {
    push(k , sl , sr);
    if (ql <= sl && sr <= qr)
    {
      tree[k].lazy += v;
      push(k , sl , sr);
      return;
    }
    if (qr < sl || sr < ql)
      return;
    int mid = (sl + sr) / 2;
    update(ql, qr, v, k << 1, sl, mid);
    update(ql, qr, v, k << 1 | 1, mid + 1, sr);
    tree[k] = merge(tree[k << 1], tree[k << 1 | 1]);
  }

};

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
      st[i][j] = max(st[i][j - 1], st[i + (1 << (j - 1))][j - 1]);
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
namespace __gnu_pbds {
typedef tree<ll,
        null_type,
        less_equal<ll>,
        rb_tree_tag,
        tree_order_statistics_node_update> ordered_set;
}
using namespace __gnu_pbds;

void Insert(ordered_set &s, ll x) { //this function inserts one more occurrence of (x) into the set.
  s.insert(x);
}

bool Exist(ordered_set &s, ll x) { //this function checks weather the value (x) exists in the set or not.
  if ((s.upper_bound(x)) == s.end()) {
    return 0;
  }
  return ((*s.upper_bound(x)) == x);
}

void Erase(ordered_set &s, ll x) { //this function erases one occurrence of the value (x).
  if (Exist(s, x)) {
    s.erase(s.upper_bound(x));
  }
}

ll FirstIdx(ordered_set &s, ll x) { //this function returns the first index of the value (x)..(0 indexing).
  if (!Exist(s, x)) {
    return -1;
  }
  return (s.order_of_key(x));
}

ll Value(ordered_set &s, ll idx) { //this function returns the value at the index (idx)..(0 indexing).
  return (*s.find_by_order(idx));
}

ll LastIdx(ordered_set &s, ll x) { //this function returns the last index of the value (x)..(0 indexing).
  if (!Exist(s, x)) {
    return -1;
  }
  if (Value(s, (int)s.size() - 1) == x) {
    return (int)(s.size()) - 1;
  }
  return FirstIdx(s, *s.lower_bound(x)) - 1;
}
ll Count(ordered_set &s, ll x) { //this function returns the number of occurrences of the value (x).
  if (!Exist(s, x)) {
    return 0;
  }
  return LastIdx(s, x) - FirstIdx(s, x) + 1;
}
void Clear(ordered_set &s) { //this function clears all the elements from the set.
  s.clear();
}

ll Size(ordered_set &s) { //this function returns the size of the set.
  return (int)(s.size());
}
ll how_many_smaller_equal(ordered_set &s , ll x) {
  auto it = s.lower_bound(x);
  if (it == s.end()) {
    return s.size();
  }
  ll idx = FirstIdx(s , *it);
  return idx;
}
//MAX FLOW
//   min node cover (each edge has atleast node in set) = max bip match
// Edmonds karp O(V E^2)
int n , m;
ll cap[501][501];
vector<int>adj[501];
int anc[501];
ll get_flow(int s , int t) {
  if (s == t)return 0;
  auto reach = [&]()->bool
  {
    memset(anc , -1 , sizeof(int) * (n + 1));
    deque<int> qu;
    qu.push_back(s);

    while (!qu.empty()) {
      int node = qu.front();
      qu.pop_front();
      if (node == t)break;
      for (auto nxt : adj[node]) {
        if (cap[node][nxt] < 1 || anc[nxt] != -1 || nxt == s)continue;
        anc[nxt] = node;
        qu.push_back(nxt);
      }
    }
    return anc[t] != -1;
  };
  ll flow = 0;
  while (reach()) {
    int cur = t;
    ll mn = INF;
    while (cur != s) {
      int par = anc[cur];
      chmin(mn , cap[par][cur]);
      cur = par;
    }
    cur = t;
    while (cur != s) {
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
//Max bipartite matchin , Max Flow any flow shit
struct Dinic {//O(V^2 . E) , max match bip.. O((V+ E) . sqrt(V))
  using F = long long; // flow type
  struct Edge {
    int to;
    F flo, cap;
  };

  int N;
  vector<Edge> eds;
  vector<vector<int>> adj;
  vector<vector<int>::iterator> cur;
  vector<int> lev;

  void init(int _N) {
    N = _N;
    adj.resize(N);
    cur.resize(N);
  }

  // void reset() { for (auto &e : eds) e.flo = 0; }

  void ae(int u, int v, F cap, F rcap = 0) {
    // Ensure capacities are nonnegative.
    assert(min(cap, rcap) >= 0);
    // Add forward edge.
    adj[u].push_back((int)eds.size());
    eds.push_back({v, 0, cap});
    // Add reverse edge.
    adj[v].push_back((int)eds.size());
    eds.push_back({u, 0, rcap});
  }
  bool bfs(int s, int t) {
    lev = vector<int>(N, -1);
    for (int i = 0; i < N; i++) {
      cur[i] = adj[i].begin();
    }
    queue<int> q;
    q.push(s);
    lev[s] = 0;
    while (!q.empty()) {
      int u = q.front();
      q.pop();
      for (auto &edgeIndex : adj[u]) {
        const Edge &E = eds[edgeIndex];
        int v = E.to;
        if (lev[v] < 0 && E.flo < E.cap) {
          q.push(v);
          lev[v] = lev[u] + 1;
        }
      }
    }
    return lev[t] >= 0;
  }
  F dfs(int v, int t, F flo) {
    if (v == t) return flo;
    for (; cur[v] != adj[v].end(); cur[v]++) {
      Edge &E = eds[*cur[v]];
      if (lev[E.to] != lev[v] + 1 || E.flo == E.cap)
        continue;
      F df = dfs(E.to, t, min(flo, E.cap - E.flo));
      if (df) {
        E.flo += df;
        eds[(*cur[v]) ^ 1].flo -= df;
        return df;
      }
    }
    return 0;
  }
  F maxFlow(int s, int t) {
    F tot = 0;
    while (bfs(s, t)) {
      while (F df = dfs(s, t, numeric_limits<F>::max()))
        tot += df;
    }
    return tot;
  }
};
//O(F E)
struct Edge {
  int u, v;
  ll cap, flow, cost;
};

struct MinCostMaxFlow {
  int n;
  vector<Edge> edges;
  vector<vector<int>> graph;
  MinCostMaxFlow(int n) : n(n), graph(n) {}
  // Add a directed edge from u to v with capacity and cost.
  // Also adds a reverse edge with zero capacity and negative cost.
  void addEdge(int u, int v, ll cap, ll cost) {
    edges.push_back({u, v, cap, 0, cost});
    edges.push_back({v, u, 0, 0, -cost});
    graph[u].push_back(edges.size() - 2);
    graph[v].push_back(edges.size() - 1);
  }

  // Computes the minimum cost maximum flow from s to t.
  // Returns a pair {total flow, total cost}.
  pair<ll, ll> minCostMaxFlow(int s, int t) {
    ll flow = 0, cost = 0;
    while (true) {
      vector<ll> dist(n, INF);
      vector<int> parent(n, -1), parentEdge(n, -1);
      vector<bool> inQueue(n, false);
      dist[s] = 0;
      queue<int> q;
      q.push(s);
      inQueue[s] = true;

      // SPFA to find shortest path from s to t.
      while (!q.empty()) {
        int u = q.front();
        q.pop();
        inQueue[u] = false;
        for (int idx : graph[u]) {
          Edge &e = edges[idx];
          if (e.cap > e.flow && dist[e.v] > dist[u] + e.cost) {
            dist[e.v] = dist[u] + e.cost;
            parent[e.v] = u;
            parentEdge[e.v] = idx;
            if (!inQueue[e.v]) {
              q.push(e.v);
              inQueue[e.v] = true;
            }
          }
        }
      }
      if (parent[t] == -1) break; // No augmenting path found.

      // Find the maximum flow we can send through the found path.
      ll push_flow = INF;
      for (int v = t; v != s; v = parent[v])
        push_flow = min(push_flow, edges[parentEdge[v]].cap - edges[parentEdge[v]].flow);

      // Augment flow along the path.
      for (int v = t; v != s; v = parent[v]) {
        int idx = parentEdge[v];
        edges[idx].flow += push_flow;
        edges[idx ^ 1].flow -= push_flow;
      }
      flow += push_flow;
      cost += push_flow * dist[t];
    }
    return {flow, cost};
  }
};
//FFT
using cd = complex<double>;
const double PI = acos(-1);

void fft(vector<cd> & a, bool invert) {
    int n = a.size();

    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;

        if (i < j)
            swap(a[i], a[j]);
    }

    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2 * PI / len * (invert ? -1 : 1);
        cd wlen(cos(ang), sin(ang));
        for (int i = 0; i < n; i += len) {
            cd w(1);
            for (int j = 0; j < len / 2; j++) {
                cd u = a[i+j], v = a[i+j+len/2] * w;
                a[i+j] = u + v;
                a[i+j+len/2] = u - v;
                w *= wlen;
            }
        }
    }

    if (invert) {
        for (cd & x : a)
            x /= n;
    }
}

vector<int> multiply(vector<int> const& a, vector<int> const& b) {
    vector<cd> fa(a.begin(), a.end()), fb(b.begin(), b.end());
    int n = 1;
    while (n < (int)a.size() + b.size())
        n <<= 1;
    fa.resize(n);
    fb.resize(n);

    fft(fa, false);
    fft(fb, false);
    for (int i = 0; i < n; i++)
        fa[i] *= fb[i];
    fft(fa, true);

    vector<int> result(n);
    for (int i = 0; i < n; i++)
        result[i] = round(fa[i].real());
    return result;
}

vector<int> poly_pow(vector<int> poly, int p, int limit = 1e9) {
    vector<int> ans{1};
    while (p) {
        if(p&1) ans = conv(ans, poly);
        poly = conv(poly, poly);
        ans.resize(limit + 1);
        poly.resize(limit + 1);
        p >>= 1;
    }
    return ans;
}
/*
*   Using Taylor Polynomial Shift + FFT/NTT with mod ..
*/
vector<int> poly_shift(const vector<int> &a , ll k){
  int N = a.size();
  vector<int> C(N , 0) , G(N + 1 , 0);
  ll k_pow = 1;
  //build C and G
  for(int i = 0;i < N;i++){
      C[i] = mult(a[i] , FACT[i]);
      G[N - i] = mult(k_pow , invFACT[i]);
      k_pow *= k;
      k_pow %= mod;
      if(k_pow < 0)k_pow += mod;
  }
  //convolution under mod
  auto cres = conv(C , G);
  vector<int>res(N);
  for(int i = 0;i < N;i++){//scaling
      res[i] = mult(cres[i + N] , invFACT[i]);
  }
  return res;
}
//FFT MOD
#define vi vector<int>
#define rep(aa, bb, cc) for(int aa = bb; aa < cc;aa++)
#define sz(a) (int)a.size()
typedef complex<double> C;
typedef vector<double> vd;
void fft(vector<C>& a) {
    int n = sz(a), L = 31 - __builtin_clz(n);
    static vector<complex<long double>> R(2, 1);
    static vector<C> rt(2, 1);  // (^ 10% faster if double)
    for (static int k = 2; k < n; k *= 2) {
        R.resize(n); rt.resize(n);
        auto x = polar(1.0L, acos(-1.0L) / k);
        rep(i,k,2*k) rt[i] = R[i] = i&1 ? R[i/2] * x : R[i/2];
    }
    vi rev(n);
    rep(i,0,n) rev[i] = (rev[i / 2] | (i & 1) << L) / 2;
    rep(i,0,n) if (i < rev[i]) swap(a[i], a[rev[i]]);
    for (int k = 1; k < n; k *= 2)
        for (int i = 0; i < n; i += 2 * k) rep(j,0,k) {
                // C z = rt[j+k] * a[i+j+k]; // (25% faster if hand-rolled)  /// include-line
                auto x = (double *)&rt[j+k], y = (double *)&a[i+j+k];        /// exclude-line
                C z(x[0]*y[0] - x[1]*y[1], x[0]*y[1] + x[1]*y[0]);           /// exclude-line
                a[i + j + k] = a[i + j] - z;
                a[i + j] += z;
            }
}
typedef vector<ll> vl;
template<int M> vl convMod(const vl &a, const vl &b) {
	if (a.empty() || b.empty()) return {};
	vl res(sz(a) + sz(b) - 1);
	int B=32-__builtin_clz(sz(res)), n=1<<B, cut=(int)(sqrt(M));
	vector<C> L(n), R(n), outs(n), outl(n);
	rep(i,0,sz(a)) L[i] = C((int)a[i] / cut, (int)a[i] % cut);
	rep(i,0,sz(b)) R[i] = C((int)b[i] / cut, (int)b[i] % cut);
	fft(L), fft(R);
	rep(i,0,n) {
		int j = -i & (n - 1);
		outl[j] = (L[i] + conj(L[j])) * R[i] / (2.0 * n);
		outs[j] = (L[i] - conj(L[j])) * R[i] / (2.0 * n) / 1i;
	}
	fft(outl), fft(outs);
	rep(i,0,sz(res)) {
		ll av = (ll)(real(outl[i])+.5), cv = (ll)(imag(outs[i])+.5);
		ll bv = (ll)(imag(outl[i])+.5) + (ll)(real(outs[i])+.5);
		res[i] = ((av % M * cut + bv) % M * cut + cv) % M;
	}
	return res;
}
vector<int> poly_pow(vector<int> poly, int p, int limit) {
    vector<int> ans{1};
    while (p) {
        if(p&1) ans = convMod<mod>(ans, poly);
        poly = convMod<mod>(poly, poly);
        if(ans.size() > limit)
            ans.resize(limit);
        if(poly.size() > limit)
            poly.resize(limit);
        p >>= 1;
    }
    return ans;
}

// NTT MOD CRT SHIT

const ll mod = (119 << 23) + 1, root = 62; // = 998244353 , generator = 3
// For p < 2^30 there is also e.g. 5 << 25, 7 << 26, 479 << 21
// and 483 << 21 (same root). The last two are > 10^9.


ll modpow(ll b, ll e) {
    ll ans = 1;
    for (; e; b = b * b % mod, e /= 2)
        if (e & 1) ans = ans * b % mod;
    return ans;
}

// Primitive Root of the mod of form 2^a * b + 1
int generator () {
    vector<int> fact;
    int phi = mod-1,  n = phi;
    for (int i=2; i*i<=n; ++i)
        if (n % i == 0) {
            fact.push_back (i);
            while (n % i == 0)
                n /= i;
        }
    if (n > 1)
        fact.push_back (n);

    for (int res=2; res<=mod; ++res) {
        bool ok = true;
        for (size_t i=0; i<fact.size() && ok; ++i)
            ok &= modpow (res, phi / fact[i]) != 1;
        if (ok)  return res;
    }
    return -1;
}
int modpow(int b, int e, int m) {
    int ans = 1;
    for (; e; b = (ll)b * b % m, e /= 2)
        if (e & 1) ans = (ll)ans * b % m;
    return ans;
}

void ntt(vector<int> &a) {
    int n = (int)a.size(), L = 31 - __builtin_clz(n);
    vector<int> rt(2, 1); // erase the static if you want to use two moduli;
    for (int k = 2, s = 2; k < n; k *= 2, s++) { // erase the static if you want to use two moduli;
        rt.resize(n);
        int z[] = {1, modpow(root, mod >> s, mod)};
        for (int i = k; i < 2*k; ++i) rt[i] = (ll)rt[i / 2] * z[i & 1] % mod;
    }
    vector<int> rev(n);
    for (int i = 0; i < n; ++i) rev[i] = (rev[i / 2] | (i & 1) << L) / 2;
    for (int i = 0; i < n; ++i) if (i < rev[i]) swap(a[i], a[rev[i]]);
    for (int k = 1; k < n; k *= 2) {
        for (int i = 0; i < n; i += 2 * k) {
            for (int j = 0; j < k; ++j) {
                int z = (ll)rt[j + k] * a[i + j + k] % mod, &ai = a[i + j];
                a[i + j + k] = ai - z + (z > ai ? mod : 0);
                ai += (ai + z >= mod ? z - mod : z);
            }
        }
    }
}
vector<int> conv(const vector<int> &a, const vector<int> &b) {
    if (a.empty() || b.empty()) return {};
    int s = (int)a.size() + (int)b.size() - 1, B = 32 - __builtin_clz(s), n = 1 << B;
    int inv = modpow(n, mod - 2, mod);
    vector<int> L(a), R(b), out(n);
    L.resize(n), R.resize(n);
    ntt(L), ntt(R);
    for (int i = 0; i < n; ++i) out[-i & (n - 1)] = (ll)L[i] * R[i] % mod * inv % mod;
    ntt(out);
    return {out.begin(), out.begin() + s};
}

ll CRT(ll a, ll m1, ll b, ll m2) {
    __int128 m = m1*m2;
    ll ans = a*m2%m*modpow(m2, m1-2, m1)%m + m1*b%m*modpow(m1, m2-2, m2)%m;
    return ans % m;
}


/*

int mod, root, desired_mod = 1000000007;
const int mod1 = 167772161;
const int mod2 = 469762049;
const int mod3 = 754974721;
const int root1 = 3;
const int root2 = 3;
const int root3 = 11;

int CRT(int a, int b, int c, int m1, int m2, int m3) {
    __int128 M = (__int128)m1*m2*m3;
    ll M1 = (ll)m2*m3;
    ll M2 = (ll)m1*m3;
    ll M3 = (ll)m2*m1;

    int M_1 = modpow(M1%m1, m1 - 2, m1);
    int M_2 = modpow(M2%m2, m2 - 2, m2);
    int M_3 = modpow(M3%m3, m3 - 2, m3);

    __int128 ans = (__int128)a*M1*M_1;
    ans += (__int128)b*M2*M_2;
    ans += (__int128)c*M3*M_3;

    return (ans % M) % desired_mod;
}

*/
__int128 read() {
  __int128 x = 0, f = 1;
  char ch = getchar();
  while (ch < '0' || ch > '9') {
      if (ch == '-') f = -1;
      ch = getchar();
  }
  while (ch >= '0' && ch <= '9') {
      x = x * 10 + ch - '0';
      ch = getchar();
  }
  return x * f;
}
void print(__int128 x) {
  if (x < 0) {
      putchar('-');
      x = -x;
  }
  if (x > 9) print(x / 10);
  putchar(x % 10 + '0');
}

// MATRIX EXPO ARRAY C WRAPPER , FASTER THAN VECTOR & HEAPS YAY!!!
const int MAX = 103;
using matrix = array<array<int, MAX>, MAX>;
 
matrix operator*(const matrix &a, const matrix &b){
	matrix ret;
	
	for (int i = 0; i < MAX; i ++)
		for (int j = 0; j < MAX; j ++)
			ret[i][j] = 0;
	
	for (int r = 0; r < 103; r ++)
		for (int c = 0; c < 103; c ++)
            {
                ll sum = 0;
                for (int k = 0; k < 103; k ++){
                    (sum += ( 1ll * a[r][k]) * (b[k][c]) )%= mod;
                    
                }
                ret[r][c] = sum;
            }

	
	return ret;
}
auto initialize = [&](auto &mt){
		for (int i = 0; i < MAX; i ++)
			for (int j = 0; j < MAX; j ++)
				mt[i][j] = 0;
	};
    //EXPONENTIATION BINARY..
for(int i = 1;i <= k;i <<= 1){
        if(i & k)T = T * trans;
        trans = trans * trans;
    }
// -----------------------------------------------

// GEOMETRY!
/*
DONT FORGET , Point USES DOUBLE IN ITS INNERS!!! NOTE FOR Long Doubles !!
*/
//#define M_PI       3.14159265358979323846   // pi
//#define double long double
#define int ll
const double PI = acosl(-1);
const double EPS = 1e-9;
template < typename T = int > struct Point {
    T x, y;
    Point(T _x = 0, T _y = 0) : x(_x), y(_y) {}
    Point(const Point &p) : x(p.x), y(p.y) {}
    Point operator + (const Point &p) const { return Point(x + p.x, y + p.y); }
    Point operator - (const Point &p) const { return Point(x - p.x, y - p.y); }
    Point operator * (T c) const { return Point(x * c, y * c); }
    Point operator / (T c) const { return Point(x / c, y / c); }
    bool operator == (const Point &p) const { return x == p.x && y == p.y; }
    bool operator != (const Point &p) const { return x != p.x || y != p.y; }
    bool operator < (const Point &p) const { return make_pair(y, x) < make_pair(p.y, p.x); }
    bool operator > (const Point &p) const { return make_pair(y, x) > make_pair(p.y, p.x); }
    bool operator <= (const Point &p) const { return make_pair(y, x) <= make_pair(p.y, p.x); }
    bool operator >= (const Point &p) const { return make_pair(y, x) >= make_pair(p.y, p.x); }
    T dot(const Point &p) const { return x * p.x + y * p.y; }
    T cross(const Point &p) const { return x * p.y - y * p.x; }
    T cross(const Point &a, const Point &b) const { return (a - *this).cross(b - *this); }
    T dist() const { return x * x + y * y; }
    T dist(const Point &p) const { return (*this - p).dist(); }
    double distance() const { return sqrt(1.0 * dist()); }
    double distance(const Point &p) const { return sqrt(1.0 * dist(p)); }
    double angle() const { return atan2l(y, x); }
    double angle(const Point &p) const { return atan2l(cross(p), dot(p)); }
    Point unit() const { return *this / distance(); }
    Point perp() const { return Point(-y, x); }
    Point rotate(double a) const { return Point(x * cos(a) - y * sin(a), x * sin(a) + y * cos(a)); }
    Point rotate(const Point &p, double a) const { return (*this - p).rotate(a) + p; }
    Point normal() const { return perp().unit(); }
    friend istream& operator >> (istream &in, Point &p) { return in >> p.x >> p.y; }
    friend ostream& operator << (ostream &out, const Point &p) { return out << '(' << p.x << ' ' << p.y << ')'; }
};
using pt = Point<double>;
using Line = pair<pt , pt>;//start point , direction
bool intersects(const Line &l1 ,const Line &l2){//you may need to call function (l1 , l2) and (l2 , l1)..
    auto [a1 , d1] = l1;
    auto [a2 , d2] = l2;
    double numo = (a2 - a1).cross(d2);
    double deno = d1.cross(d2);
    if(fabs(deno) < EPS){
        //= zero , lines are parallel or same line , impossible in our problem
        return false;
    }
    double t = numo / deno;
    if(t < -EPS || t > 1 + EPS)return false;
    return true;
}
template < typename T = int > struct Line {
    pt v; T c;
    // From direction vector v and offset c
    Line(pt v, T c) : v(v), c(c) {}
    // From equation ax+by=c
    Line(T a, T b, T c) : v({b,-a}), c(c) {}
    // From points P and Q
    Line(pt p, pt q) : v(q-p), c(v.cross(p)) {}
    // Will be defined later:
    // - these work with T = int
    T side(pt p) {return v.cross(p)-c;}
    double dist(pt p) {return abs(side(p)) / v.distance();}
    Line perpThrough(pt p) {return {p, p + v.perp()};}
    bool cmpProj(pt p, pt q) {return v.dot(p) < v.dot(q);}//sort points on line , along direction ..
    Line translate(pt t) {return {v, c + v.cross(t)};}
    // - these require T = double
    Line shiftLeft(double dist) {return {v, c + dist*(v.distance())};}
    pt proj(pt p) {return p - v.perp()*side(p)/(v.dist());}
    pt refl(pt p) {return p - v.perp()*2*side(p)/(v.dist());}
};
using line = Line<double>;
//line - line intersection
bool inter(line l1, line l2, pt &out) {
    double d = l1.v.cross(l2.v);
    if (d == 0) return false;
    out = (l2.v*l1.c - l1.v*l2.c) / d; // requires floating-point coordinates
    return true;
}
line bisector(line l1, line l2, bool interior) {
    assert(l1.v.cross(l2.v) != 0); // l1 and l2 cannot be parallel!
    double sign = interior ? 1 : -1;
    return {l2.v/(l2.v.distance()) + l1.v/(l1.v.distance()) * sign,
    l2.c/(l2.v.distance()) + l1.c/(l1.v.distance()) * sign};
}
bool inDisk(pt a, pt b, pt p) {
    return (a-p).dot(b-p) <= 0;
}
int orient(pt a, pt b, pt c) {return (b-a).cross(c-a);}
bool onSegment(pt a , pt b , pt p){
    return orient(a , b , p) == 0 && inDisk(a , b, p);
}
// intersection between segments A---B and C---D
bool properInter(pt a, pt b, pt c, pt d, pt &out) {
  double oa = orient(c,d,a),
  ob = orient(c,d,b),
  oc = orient(a,b,c),
  od = orient(a,b,d);
  // Proper intersection exists iff opposite signs
  if (oa*ob < 0 && oc*od < 0) {
    out = (a*ob - b*oa) / (ob-oa);
    return true;
  }
  return false;
}
// To create sets of points we need a comparison function
struct cmpX {
    bool operator()(pt a, pt b) const {
        return make_pair(a.x, a.y) < make_pair(b.x, b.y);
    }
};
set<pt,cmpX> inters(pt a, pt b, pt c, pt d) {
    pt out;
    if (properInter(a,b,c,d,out)) return {out};
    set<pt,cmpX> s;
    if (onSegment(c,d,a)) s.insert(a);
    if (onSegment(c,d,b)) s.insert(b);
    if (onSegment(a,b,c)) s.insert(c);
    if (onSegment(a,b,d)) s.insert(d);
    return s;
}
// dist from p to seg a--b
double segPoint(pt a, pt b, pt p) {
  if (a != b) {
    line l(a,b);
    if (l.cmpProj(a,p) && l.cmpProj(p,b)) // if closest to projection
      return l.dist(p); // output distance to line
  }
  return min((p-a).distance(), (p-b).distance()); // otherwise distance to A or B
}
// dist between 2 segments..
double segSeg(pt a, pt b, pt c, pt d) {
  pt dummy;
  if (properInter(a,b,c,d,dummy))
    return 0;
  return min({segPoint(a,b,c), segPoint(a,b,d),
    segPoint(c,d,a), segPoint(c,d,b)});
}
bool half(pt p) { // true if in blue half
    assert(p.x != 0 || p.y != 0); // the argument of (0,0) is undefined
    return p.y > 0 || (p.y == 0 && p.x < 0);
}
void polarSort(vector<pt> &v) {
    sort(v.begin(), v.end(), [](pt v, pt w) {
        return make_tuple(half(v), 0) < make_tuple(half(w), v.cross(w));
    });
}

// Ray Test , are we inside a polygon?
// true if P at least as high as A (blue part)
bool above(pt a, pt p) {
  return p.y >= a.y;
}
// check if [PQ] crosses ray from A
bool crossesRay(pt a, pt p, pt q) {
  return (above(a,q) - above(a,p)) * orient(a,p,q) > 0;
}
// if strict, returns false when A is on the boundary
bool inPolygon(vector<pt> p, pt a, bool strict = true) {
  int numCrossings = 0;
  for (int i = 0, n = p.size(); i < n; i++) {
    if (onSegment(p[i], p[(i+1)%n], a))
      return !strict;
    numCrossings += crossesRay(a, p[i], p[(i+1)%n]);
  }
  return numCrossings & 1; // inside if odd number of crossings
}
// accumulate non-negative doubles for less absolute error
struct stableSum {
  int cnt = 0;
  vector<double> v, pref{0};
  void operator+=(double a) {
    assert(a >= 0);
    int s = ++cnt;
    while (s % 2 == 0) {
      a += v.back();
      v.pop_back(), pref.pop_back();
      s /= 2;
    }
    v.push_back(a);
    pref.push_back(pref.back() + a);
  }
  double val() {return pref.back();}
};
//Circles Stuff , Copied , not tested
// Sign function: returns +1 if x>EPS, -1 if x<-EPS, else 0
inline int sgn(ld x) { return (x > EPS) - (x < -EPS); }

// Circle represented by center o and radius r
struct Circle {
    pt o;
    ld r;
    Circle() : o(pt()), r(0) {}
    Circle(pt _o, ld _r) : o(_o), r(_r) {}
};

// Check if point p lies inside or on circle C
// Idea: compare squared distance to squared radius for stability
bool contains(const Circle& C, const Point& p) {
    return sgn(dist2(C.o, p) - C.r*C.r) <= 0;
}

// Intersection of circle C with line through points A -> B
// Idea: project center onto line, then find distances along perpendicular
vector<Point> circleLineIntersection(const Circle& C, const Point& A, const Point& B) {
    Point dir = B - A;
    // Projection parameter t of center onto the line 
    ld t = dot(C.o - A, dir) / norm2(dir);
    Point proj = A + dir * t;
    // Distance^2 from projection to intersection points
    ld h2 = C.r*C.r - dist2(proj, C.o);
    vector<Point> res;
    if (sgn(h2) < 0) return res;                  // no intersection
    ld h = sqrt(max((ld)0, h2));                // perpendicular offset
    Point unit = dir / sqrt(norm2(dir));        // unit direction of line
    res.push_back(proj + unit * h);
    if (sgn(h) != 0)
        res.push_back(proj - unit * h);
    return res;
}

// Intersection points of two circles C1 and C2
// Geometric concept: two circles intersect in 0, 1, or 2 points.
// We find the line of intersection (the radical line) by dropping a perpendicular
// from C1.o towards C2.o at distance x = (d^2 + r1^2 - r2^2)/(2d). Then the intersection points
// lie at a perpendicular distance h = sqrt(r1^2 - x^2) from that foot point.
vector<Point> circleCircleIntersection(const Circle& C1, const Circle& C2) {
    ld d = dist(C1.o, C2.o);
    // Check for infinite or no solutions
    if (sgn(d) == 0 && sgn(C1.r - C2.r) == 0) return {};
    if (sgn(d - C1.r - C2.r) > 0) return {};
    if (sgn(d - fabs(C1.r - C2.r)) < 0) return {};
    // Distance from C1.o to the line connecting intersection points
    ld x = (d*d - C2.r*C2.r + C1.r*C1.r) / (2*d);
    ld h2 = C1.r*C1.r - x*x;
    Point v = (C2.o - C1.o) / d;
    Point p = C1.o + v * x;                      // foot of perpendicular on radical line
    if (sgn(h2) < 0) return {p};                // one intersection (tangent)
    ld h = sqrt(h2);
    // Intersection points by moving ± perpendicular to v
    return { p + Point(-v.y, v.x) * h,
             p - Point(-v.y, v.x) * h };
}

// Tangent points from external point P to circle C
// Geometric concept: tangents form right angles with the radius at the contact point.
// Construct right triangle: OP of length d, radius r. Distance from O to tangent foot is l = r^2/d,
// and offset distance h = sqrt(d^2 - l^2). Direction is along OP, offset perpendicular.
vector<Point> tangents(const Circle& C, const Point& P) {
    vector<Point> res;
    Point v = P - C.o;
    ld d2 = norm2(v);
    ld r2 = C.r*C.r;
    if (d2 < r2 - EPS) return res;              // P inside, no tangents
    ld d = sqrt(d2);
    ld l = r2 / d;                              // distance from O to tangent base
    ld h2 = d2 - l*l;                           // squared distance from base to tangent points
    Point u = v / d;
    Point perp(-u.y, u.x);
    // Two tangent points: base plus/minus perpendicular component
    res.push_back(C.o + u * l + perp * sqrt(max((ld)0, h2)));
    if (sgn(h2) != 0)
        res.push_back(C.o + u * l - perp * sqrt(max((ld)0, h2)));
    return res;
}

// Area of overlap between two circles A and B
// Geometric concept: overlapping region is union of two circular segments.
// For partial overlap, compute segment angles α, β via law of cosines:
// α = 2*acos((d^2 + r1^2 - r2^2)/(2*d*r1)), similarly β. Then area =
// 0.5*r1^2*(α - sinα) + 0.5*r2^2*(β - sinβ).
ld circleOverlapArea(const Circle& A, const Circle& B) {
    ld d = A.o.distance(B.o);
    // No overlap
    if (sgn(d - (A.r + B.r)) >= 0) return 0;
    // One circle completely inside the other
    if (sgn(d + min(A.r, B.r) - max(A.r, B.r)) <= 0)
        return PI * min(A.r, B.r) * min(A.r, B.r);
    // Partial overlap: compute two segment areas
    ld x = (d*d + A.r*A.r - B.r*B.r) / (2*d);
    // Central angles for segments
    ld ang1 = 2 * acosl(x / A.r);
    ld ang2 = 2 * acosl((d-x) / B.r);
    // Segment areas
    ld area1 = 0.5 * A.r*A.r * (ang1 - sinl(ang1));
    ld area2 = 0.5 * B.r*B.r * (ang2 - sinl(ang2));
    return area1 + area2;
}

// Example usage:
// Circle C(Point(0,0), 5);
// vector<Point> pts = circleCircleIntersection(C, Circle(Point(5,0), 5));
// for (auto &p : pts) cout << p.x << " " << p.y << "\n";

struct Convex_Hull {
    // cross = (b−a) × (c−a)
    static ll cross(const pt& a, const pt& b, const pt& c) {
        return (b.x - a.x)*(c.y - a.y)
             - (b.y - a.y)*(c.x - a.x);
    }

    // orientation using cross
    //  >0: left turn (CCW), <0: right turn (CW), 0: collinear
    int orientation(const pt& a, const pt& b, const pt& c) {
        ll v = cross(a,b,c);
        if (v < 0) return -1;
        if (v > 0) return  1;
        return 0;
    }

    // true if a→b→c is a left turn (or collinear, if include_collinear)
    bool ccw(const pt& a, const pt& b, const pt& c, bool include_collinear) {
        int o = orientation(a,b,c);
        return o > 0 || (include_collinear && o == 0);
    }

    // true if a,b,c are collinear
    bool is_collinear(const pt& a, const pt& b, const pt& c) {
        return orientation(a,b,c) == 0;
    }

    vector<pt> convex_points;

    // points: input array (will be re‐ordered)
    // include_collinear: if true, keeps collinear pts on hull edges
    Convex_Hull(vector<pt>& points, bool include_collinear = false) {
        // 1) find pivot
        pt p0 = *min_element(points.begin(), points.end(),
                             [&](auto &a, auto &b){
                                 return a.y < b.y || (a.y==b.y && a.x < b.x);
                             });
        // 2) sort by angle CCW around p0
        sort(points.begin(), points.end(), [&](auto &a, auto &b){
            int o = orientation(p0, a, b);
            if (o == 0) 
                return (p0.dist(a) < p0.dist(b));  // closer first
            return o > 0;  // left turn (a before b) for CCW
        });
        // 3) optionally reverse the tail to keep farthest collinear last
        if (include_collinear) {
            int i = (int)points.size()-1;
            while (i > 0 && is_collinear(p0, points[i-1], points.back())) 
                --i;
            reverse(points.begin()+i, points.end());
        }
        // 4) build hull
        for (auto &pt : points) {
            while (convex_points.size() >= 2
                && !ccw(convex_points[convex_points.size()-2],
                        convex_points.back(),
                        pt,
                        include_collinear))
            {
                convex_points.pop_back();
            }
            convex_points.push_back(pt);
        }
    }
};
//take care of "strictly of in / out stuff"
bool inConvexPolygon(const vector<pt> & vec , pt p){//good for convex polys CCW O(log n)
	if(vec.size() == 0)return false;
    if(vec.size() == 1)return p == vec.back();//point
    if(vec.size() == 2)return onSegment(vec[0] , vec[1] , p);//line
    int n = vec.size();
    if(orient(vec[0] , vec[1] , p) < 0)return false;//out of widge..
    if(orient(vec[0] , vec[n - 1] , p) > 0)return false;
    int tar = -1;
    int l = 1 , r = n - 2 , mid;
    while(l <= r){
        mid = l + (r - l)/2;
        if(orient(vec[0] , vec[mid] , p) >= 0){
            tar = mid;
            l = mid + 1;
        }
        else r = mid - 1;
    }
    return orient(vec[tar] , vec[tar + 1] , p) >= 0;
}
// WINDING Numbers
// returns >0 for p left of AB, =0 for on AB, <0 for p right of AB
double isLeft(const Point& A, const Point& B, const Point& P) {
    return (B.x - A.x) * (P.y - A.y) - (P.x - A.x) * (B.y - A.y);
}

// winding number test: returns how many times polygon winds around P
// >0 means inside (counter‐clockwise winding), 0 means outside
int windingNumber(const vector<Point>& poly, const Point& P) {
    int wn = 0;
    int n = poly.size();
    for (int i = 0; i < n; ++i) {
        const Point& A = poly[i];
        const Point& B = poly[(i+1)%n];
        if (A.y <= P.y) {
            // upward crossing
            if (B.y > P.y && isLeft(A,B,P) > 0)
                ++wn;
        } else {
            // downward crossing
            if (B.y <= P.y && isLeft(A,B,P) < 0)
                --wn;
        }
    }
    return wn;
}

// DnC  closes Pair of points
// 2 implementations 
//Imp 1:
# 版權聲明： 本網誌所有文章除特別聲明外，均採用 (CC)BY-NC-SA 許可協議。轉載請註明出處！

double ds(point& a, point& b)
{
  clc dst..
}
double Combine(int& a, int& b, int& mid, double& l, double& r)
{
	double d = min(l, r);
	double line = (V[mid].x + V[mid + 1].x) / 2;
	double Min = d;
	for (int i = mid + 1; i <= b && V[i].x < line + d; ++i)
		for (int j = mid; j >= a && V[j].x > line - d; --j)
			Min = min(Min, ds(V[i], V[j]));
	return Min;
}
double Divide(int a, int b)
{
	if (a >= b) return 1000000;
	int mid = (a + b) / 2;
	double l = Divide(a, mid);
	double r = Divide(mid + 1, b);
	return Combine(a, b, mid, l, r);
}
int main()
{
	while (cin >> N)
	{
		read();
		sort(V.begin(), V.end(), [](point& a, point& b) { return a.x < b.x; });
		cout << Divide(0, N - 1) << "\n";
	}
}

//Imp 2
struct Point
{
	long double x, y;
	
	bool operator<(const Point &other)
	{
		if (x == other.x)
		{
			return y < other.y;
		}
		return x < other.x;
	}
};

const pair<Point, Point> INF_P{{-1e9, -1e9}, {1e9, 1e9}};

long double dist(const pair<Point, Point> &a)
{
	long double d1 = a.first.x - a.second.x;
	long double d2 = a.first.y - a.second.y;
	return sqrt(d1 * d1 + d2 * d2);
}

pair<Point, Point> get_closest_points(const pair<Point, Point> &a,
									  const pair<Point, Point> &b)
{
	return dist(a) < dist(b) ? a : b;
}

/**
 * Brute force for points with near
 * the median point in the sorted array
 */
pair<Point, Point> strip_solve(vector<Point> &points)
{
	pair<Point, Point> ans = INF_P;
	for (int i = 0; i < (int)points.size(); i++)
	{
		for (int j = i + 1; j < (int)points.size() && j - i < 9; j++)
		{
			ans = get_closest_points(ans, {points[i], points[j]});
		}
	}
	return ans;
}

/** Solve the problem for range [l, r] */
pair<Point, Point> solve_closest_pair(vector<Point> &points, int l, int r)
{
	if (l == r)
	{
		return INF_P;
	}
	int mid = (l + r) / 2;

	// The smallest distance in range [l, mid]
	pair<Point, Point> ans_left = solve_closest_pair(points, l, mid);
	// The smallest distance in range [mid+1, r]
	pair<Point, Point> ans_right = solve_closest_pair(points, mid + 1, r);
	pair<Point, Point> ans;

	ans = get_closest_points(ans_left, ans_right);
	long double delta = dist(ans);

	Point mid_point = points[mid];
	vector<Point> strip;
	for (int i = l; i < r; i++)
	{
		if (abs(points[i].x - mid_point.x) <= delta)
		{
			strip.push_back(points[i]);
		}
	}
	sort(strip.begin(), strip.end(),
		 [](Point a, Point b)
		 { return a.y < b.y || (a.y == b.y && a.x < b.x); });
	return get_closest_points(ans, strip_solve(strip));
}
// Line sweep , first line intersection: like bently-ottman , more like shano - ...
struct Point{
	ll x , y;
	Point operator - (const Point &p) const { return {x - p.x, y - p.y}; }
	ll cross(const Point &o)const {return x * o.y - y * o.x;}
};
bool same(const Point& a ,const Point& b){
	return abs(a.x - b.x) <= EPS && abs(a.y - b.y) <= EPS;
}
struct segment{
	Point p , q;
	int idx;
	segment(Point _p , Point _q , int idx_): idx(idx_) , p(_p) , q(_q){} 
	segment(){}
	double getY(double x) const {
		if(p.x == q.x){
			//vertical segment..
			return p.y;
		}
		return p.y + 1.0 * (x - p.x)/(q.x - p.x) * (q.y - p.y); 
	}
	bool operator<( const segment &B)const {
		// 100 WAs take care!
		double x = max(min(p.x,q.x), min(B.p.x,B.q.x));
		double yA = getY(x), yB = B.getY(x);
		if (abs(yA - yB) > EPS) 
			return yA < yB;
		// tie‐break by unique id
			return idx < B.idx;
	}
};

int orient(Point a , Point b , Point c){
	auto res = (b - a).cross (c - a);
	if((res) == 0)return 0;
	return res < 0? -1 : 1;
}
bool intersect1D(ll l1 , ll r1 , ll l2 , ll r2){
	if(r1 < l1)swap(r1 , l1);
	if(r2 < l2)swap(r2 , l2);
	return max(l1 , l2) <= min(r1 , r2);
}
bool intersects(const segment& a , const segment &b){
	return intersect1D(a.p.x , a.q.x , b.p.x , b.q.x) && intersect1D(a.p.y , a.q.y , b.p.y , b.q.y)
		&& orient(a.p , a.q , b.p) * orient(a.p , a.q , b.q) <= 0 && orient(b.p , b.q , a.p) * orient(b.p , b.q , a.q) <= 0;
}
struct event{
	Point p;
	int type , idx;
	bool operator<(const event &o)const{
		if(p.x != o.p.x)return p.x < o.p.x;
		return type > o.type;
	}
};
vector<segment>segs;
vector<event>events;
int ottman(){
	sort(all(events));
	set<segment>st;
	int fs = -1, ss = -1;
	auto below = [&](set<segment>::iterator it)
	{
		if(it == st.begin())return st.end();
		return --it;
	};
	for(auto cur : events){
		if(cur.type == 1){
			//entry
			auto [it , statts] = st.insert(segs[cur.idx]);
			auto prv = below(it);
			auto nxt = next(it);
			if(prv != st.end() && intersects(*it , *prv)){
				fs = it->idx;
				ss = prv->idx;
				break;
			}
			if(nxt != st.end() && intersects(*it , *nxt)){
				fs = it->idx;
				ss = nxt->idx;
				break;
			}
		}else{
			auto it = st.find(segs[cur.idx]);
			auto prv = below(it);
			auto nxt = next(it);
			if(prv != st.end() && nxt != st.end() && intersects(*nxt , *prv)){
				fs = nxt->idx;
				ss = prv->idx;
				break;
			}
			st.erase(it);
		}
	}
	return ..
}
// Line sweep: Bently-ottman (n + k)log n  
//FULL CODE SAMPLE , not tested..
// ----------------------------------------------------------------------------
//  Point + Segment definitions
// ----------------------------------------------------------------------------
struct Point {
    double x, y;
    bool operator<(Point const& o) const {
        if (x != o.x) return x < o.x;
        return y < o.y;
    }
    bool operator==(Point const& o) const {
        return fabs(x - o.x) < 1e-9 && fabs(y - o.y) < 1e-9;
    }
};

struct Segment {
    Point p, q;     // endpoints, not necessarily in order
    int id;         // unique identifier for tie-breaking
    // compute y at vertical line X
    double y_at(double X) const {
        if (fabs(p.x - q.x) < 1e-9) 
            return p.y;
        return p.y + (q.y - p.y) * ( (X - p.x) / (q.x - p.x) );
    }
};

// ----------------------------------------------------------------------------
//  Global sweep-line state & comparator for status structure
// ----------------------------------------------------------------------------
double sweep_x;
struct Cmp {
    bool operator()(Segment const* a, Segment const* b) const {
        double ya = a->y_at(sweep_x);
        double yb = b->y_at(sweep_x);
        if (fabs(ya - yb) > 1e-9) 
            return ya < yb;
        return a->id < b->id;
    }
};

// ----------------------------------------------------------------------------
//  Event queue: left endpoint, intersection, right endpoint
// ----------------------------------------------------------------------------
enum EventType { LEFT = 0, INTER = 1, RIGHT = 2 };

struct Event {
    double x;
    Point   pt;       // location of event
    EventType type;
    Segment *s1, *s2; // for INTER: both segments; else s1 is the segment
    bool operator<(Event const& o) const {
        if (x != o.x) return x > o.x;        // min-heap on x
        return type > o.type;                // LEFT < INTER < RIGHT
    }
};

// ----------------------------------------------------------------------------
//  Compute intersection (if any) of two segments A & B.
//  Returns true + writes intersection to 'out' if they properly intersect.
// ----------------------------------------------------------------------------
bool segIntersect(Segment const& A, Segment const& B, Point &out) {
    // parametric form: A.p + t*(A.q-A.p) == B.p + u*(B.q-B.p)
    double x1 = A.p.x, y1 = A.p.y;
    double x2 = A.q.x, y2 = A.q.y;
    double x3 = B.p.x, y3 = B.p.y;
    double x4 = B.q.x, y4 = B.q.y;
    double den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4);
    if (fabs(den) < 1e-9) return false;   // parallel or colinear
    double t = ( (x1-x3)*(y3-y4) - (y1-y3)*(x3-x4) ) / den;
    double u = ( (x1-x3)*(y1-y2) - (y1-y3)*(x1-x2) ) / den;
    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) {
        out.x = x1 + t*(x2-x1);
        out.y = y1 + t*(y2-y1);
        return true;
    }
    return false;
}

// ----------------------------------------------------------------------------
//  Main sweep: build event-queue, process, and collect intersections
// ----------------------------------------------------------------------------
vector<Point> findAllIntersections(vector<Segment>& segs) {
    priority_queue<Event> pq;
    int n = segs.size();

    // 1) populate PQ with segment endpoints
    for (int i = 0; i < n; i++) {
        auto &s = segs[i];
        // ensure p.x <= q.x
        if (s.q < s.p) swap(s.p, s.q);
        s.id = i;
        pq.push({s.p.x, s.p, LEFT,  &s, nullptr});
        pq.push({s.q.x, s.q, RIGHT, &s, nullptr});
    }

    set<Segment*, Cmp> status;
    vector<Point> result;

    auto schedule = [&](Segment* A, Segment* B) {
        if (!A || !B) return;
        Point ip;
        if (!segIntersect(*A, *B, ip)) return;
        // don't reschedule if behind sweep line
        if (ip.x < sweep_x - 1e-9) return;
        pq.push({ip.x, ip, INTER, A, B});
    };

    while (!pq.empty()) {
        Event ev = pq.top(); pq.pop();
        sweep_x = ev.x;

        if (ev.type == LEFT) {
            // insert and check its two neighbors
            auto it = status.insert(ev.s1).first;
            if (it != status.begin()) {
                schedule(*prev(it), *it);
            }
            if (next(it) != status.end()) {
                schedule(*it, *next(it));
            }
        }
        else if (ev.type == RIGHT) {
            // remove and check former neighbors
            auto it = status.find(ev.s1);
            if (it == status.end()) continue;
            auto above = next(it), below = (it==status.begin() ? status.end() : prev(it));
            if (above!=status.end() && below!=status.end()) {
                schedule(*below, *above);
            }
            status.erase(it);
        }
        else { // INTERSECTION
            result.push_back(ev.pt);
            // swap the two segments in the status:
            auto a = ev.s1, b = ev.s2;
            auto ita = status.find(a);
            auto itb = status.find(b);
            if (ita==status.end() || itb==status.end()) continue;
            // ensure ita comes before itb
            if (Cmp()(b,a)) swap(ita, itb);
            status.erase(ita);
            status.erase(itb);
            status.insert(b);
            status.insert(a);
            // after swap, check new neighbors
            auto it_new_b = status.find(b);
            if (it_new_b!=status.begin())
                schedule(*prev(it_new_b), *it_new_b);
            auto it_new_a = status.find(a);
            if (next(it_new_a)!=status.end())
                schedule(*it_new_a, *next(it_new_a));
        }
    }

    // dedupe identical points
    sort(result.begin(), result.end());
    result.erase(unique(result.begin(), result.end()), result.end());
    return result;
}

// ----------------------------------------------------------------------------
//  Driver: read n, segments, run sweep, print all intersections
// ----------------------------------------------------------------------------
int main(){
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;
    vector<Segment> segs(n);
    for (int i = 0; i < n; i++) {
        cin >> segs[i].p.x >> segs[i].p.y
            >> segs[i].q.x >> segs[i].q.y;
    }

    auto pts = findAllIntersections(segs);
    cout << pts.size() << "\n";
    cout << fixed << setprecision(6);
    for (auto &p : pts) {
        cout << p.x << " " << p.y << "\n";
    }
    return 0;
}

// EPS tester
long double x = 10000;
cerr<< (x+EPS == x)<<endl;
long double y = nextafterl(x, x+ 5);  // the smallest long double > 1.0L
long double eps = y - x;
std::cout << std::fixed << std::setprecision(20) << eps<<" , "<<x << endl;

// Interval union Segment tree on compressed Y points..
vector<ll>ys;
struct SegTree {
  int n;
  vector<int>len , cover;

  void build(int n_) {
    if (__builtin_popcount(n_) != 1)
      n = 1 << (__lg(n_) + 1);
    else
      n = n_;
    len.resize(n << 1, 0);
    cover.resize(n << 1 , 0);
  }
  int merge(int l, int r) {
    return l + r;
  }
  void push(int k , int l , int r){
	if(cover[k] > 0){
		len[k] = ys[r] - ys[l];
	}else{
		len[k] = 0;
		if((k << 1) < len.size()){
			len[k] = len[k << 1] + len[k << 1 | 1];
		}
	}
  }
  //point update
  void update(int ql, int qr, int v, int k, int sl, int sr) {
    if (ql <= sl && sr <= qr)
      {
		cover[k] += v;
		push(k , sl , sr);
		return;
	  }
    if (qr <= sl || sr <= ql)
      return;
    int mid = (sl + sr) / 2;
    update(ql, qr, v, k << 1, sl, mid);//half open interval seg tree..
    update(ql, qr, v, k << 1 | 1, mid , sr);
    push(k , sl , sr);
  }
};

// 2D seg tree
struct SegTree2D {
  int n, m;
  vector<vector<int>> tree;

  // Merge function
  int merge(int a, int b) {
    return (a + b);
  }

  // Build the outer and inner segment trees  (n . m)
  void build(int n_, int m_, vector<vector<int>>& mat) {
    // normalize sizes to next power of 2
    n = (__builtin_popcount(n_) == 1 ? n_ : 1 << (__lg(n_) + 1));
    m = (__builtin_popcount(m_) == 1 ? m_ : 1 << (__lg(m_) + 1));
    tree.assign(n << 1, vector<int>(m << 1, 0));

    // copy initial matrix into leaves
    for (int i = 0; i < n_; ++i)
      for (int j = 0; j < m_; ++j)
        tree[i + n][j + m] = mat[i][j];

    // build inner trees
    for (int i = 0; i < (n << 1); ++i)
      for (int j = m - 1; j > 0; --j)
        tree[i][j] = merge(tree[i][j << 1], tree[i][j << 1 | 1]);

    // build outer tree
    for (int i = n - 1; i > 0; --i)
      for (int j = 0; j < (m << 1); ++j)
        tree[i][j] = merge(tree[i << 1][j], tree[i << 1 | 1][j]);
  }

  // Query helper for 1D range in column
  int query_y(int x, int y1, int y2) {
    int res = 0;
    for (y1 += m, y2 += m; y1 <= y2; y1 >>= 1, y2 >>= 1) {
      if (y1 & 1) res = merge(res, tree[x][y1++]);
      if (!(y2 & 1)) res = merge(res, tree[x][y2--]);
    }
    return res;
  }

  // Query 2D range [x1..x2][y1..y2] (log n . log m)
  int query(int x1, int y1, int x2, int y2) {
    int res = 0;
    for (x1 += n, x2 += n; x1 <= x2; x1 >>= 1, x2 >>= 1) {
      if (x1 & 1) res = merge(res, query_y(x1++, y1, y2));
      if (!(x2 & 1)) res = merge(res, query_y(x2--, y1, y2));
    }
    return res;
  }

  // Point update at (x, y) with value v (log n . log m)
  void update(int x, int y, int v) {
    x += n;
    y += m;
    tree[x][y] = v;

    // update column segment tree
    for (int j = y >> 1; j > 0; j >>= 1)
      tree[x][j] = merge(tree[x][j << 1], tree[x][j << 1 | 1]);

    // update row segment trees
    for (int i = x >> 1; i > 0; i >>= 1) {
      int yy = y;
      for (int j = yy; j > 0; j >>= 1)
        tree[i][j] = merge(tree[i << 1][j], tree[i << 1 | 1][j]);
    }
  }
};

// Sack HLD dsu on trees , small to large ..blah blah
int c[OL];
int sz[OL];
int big[OL];
vector<int>adj[OL];
int freq[OL];
ll ans[OL];
ll sum[OL];
int mx = 0;

void pre(int u, int p){
    sz[u] = 1;
    for (int v : adj[u]) if(v != p){
        pre(v, u);
        sz[u] += sz[v];
        if(big[u] == 0 || sz[v] > sz[big[u]])
            big[u] = v;
    }
}
 
void upd(int col, int d){
	if(freq[col] + d > mx)mx++;
	else if(freq[col] == mx && sum[mx] == col)mx--;
	sum[freq[col]] -= col;
	freq[col] += d;
	sum[freq[col]] += col;
}
 
void collect(int u, int p, int d){
    upd(c[u], d);
    for(int v : adj[u]) if(v!=p){
        collect(v, u ,d);
    }
}
 
void dfs(int u, int p, bool keep){
    for(int v: adj[u]) if(v != p && v != big[u]){
        dfs(v, u, false);
    }
    // add to DS
    if(big[u] != 0)
        dfs(big[u], u, true);
    upd(c[u], 1);
    for(int v: adj[u]) if(v != p && v != big[u])
            collect(v, u, +1); // light/small subtrees
 
    // answer queries
    ans[u] = sum[mx];
 
    // remove from DS
    if(!keep)
        collect(u, p, -1);
}

vector<int> manacher(string s)// or &s according to your needs!
{
	string ns = "#";
	for(auto c : s){
		ns += c;
		ns += '#';
	}
	swap(s , ns);
	int n = s.size();
	s = "@" + s + "$";
	vector<int> len(n + 1);
	int l = 1, r = 1;
	for (int i = 1; i <= n; i++)
	{
		len[i] = min(r - i, len[l + (r - i)]);
		while (s[i - len[i]] == s[i + len[i]])
			len[i]++;
		if (i + len[i] > r)
		{
			l = i - len[i];
			r = i + len[i];
		}
	}
	return len;
}
// ONLINE MANACHER!!!
template <int delta> struct ManacherBase{
private:
		static const int maxn=1e5+1;
        int r[maxn];
        char s[maxn];
        int mid,n,i;
 
public:
        ManacherBase():mid(0),i(0),n(1) 
        {
        	memset(r,-1,sizeof(int)*maxn);
        	s[0]='$';
        	r[0]=0;
        }
 
        int get(int pos)
        {
        		pos++;
                if(pos<=mid)
                        return r[pos];
                else
                        return min(r[mid - (pos - mid)], n - pos - 1);
        }
 
        void addLetter(char c)
        {
                s[n]=s[n+1]=c;
 
                while(s[i - r[i] - 1 + delta] != s[i + r[i] + 1])
                        r[++i] = get(i-1);
                r[mid=i]++, n++;
        }
 
        int maxPal()
        {
                return ( n - mid - 1 ) * 2 + 1 - delta;
        }
} ;
 
struct Manacher{
private:
        ManacherBase<1> manacherEven;
        ManacherBase<0> manacherOdd;
public:
        void addLetter(char c)
        {
                manacherEven.addLetter(c);
                manacherOdd.addLetter(c);
        }
 
        int maxPal()
        {
                return max(manacherEven.maxPal(), manacherOdd.maxPal());
        }
        int getRad(int type,int pos)
        {
                if(type)
                        return manacherOdd.get(pos);
                else
                        return manacherEven.get(pos);
        }
} ;
