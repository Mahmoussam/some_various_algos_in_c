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
const int N = 1550 + 3;
int pw1[N], pw2[N], inv1[N], inv2[N];
void Hash(int base1 = 59, int base2 = 69) {
  pw1[0] = inv1[0] = 1;
  int mininv1 = binpow(base1, mod - 2, mod);
  for (int i = 1; i < N; ++i) {
    pw1[i] = mult(pw1[i - 1], base1, mod);
    inv1[i] = mult(inv1[i - 1], mininv1, mod);
  }

  pw2[0] = inv2[0] = 1;
  int mininv2 = binpow(base2, mod - 2, mod);
  for (int i = 1; i < N; ++i) {
    pw2[i] = mult(pw2[i - 1], base2, mod);
    inv2[i] = mult(inv2[i - 1], mininv2, mod);
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
      hv1 = add(hv1, mult(pw1[i], c, mod), mod);
      hv2 = add(hv2, mult(pw2[i], c, mod), mod);
      pre1[i] = hv1, pre2[i] = hv2;
    }
  }

  pair<int, int>get_hash_range(int L, int R) {
    if (!L)
      return {pre1[R], pre2[R]};
    else {
      return {(mult(inv1[L], add(pre1[R], -pre1[L - 1], mod), mod) + mod) % mod,
              (mult(inv2[L], add(pre2[R], -pre2[L - 1], mod), mod) + mod) % mod};
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
      for (int k = 0; k < c1; ++k) {
        res[i][j] = add(res[i][j], mult(a[i][k], b[k][j], mod), mod);
        //res[i][j] = res[i][j] + a[i][k] * b[k][j];
      }
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
struct Dinic {
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

// GEOMETRY!
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
    int d = l1.v.cross(l2.v);
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
    bool operator()(pt a, pt b) {
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

bool half(pt p) { // true if in blue half
    assert(p.x != 0 || p.y != 0); // the argument of (0,0) is undefined
    return p.y > 0 || (p.y == 0 && p.x < 0);
}
void polarSort(vector<pt> &v) {
    sort(v.begin(), v.end(), [](pt v, pt w) {
        return make_tuple(half(v), 0) < make_tuple(half(w), v.cross(w));
    });
}
