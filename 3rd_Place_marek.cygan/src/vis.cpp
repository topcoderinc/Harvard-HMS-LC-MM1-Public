#include <string>
std::string DATA_PATH = "example_extracted/";

int SHIFT = 0;
int VISUALIZE = 0;
int PIECES = -1;
int ROTATE = 0;

const int MAX_PID = 700;
const int MAX_SID = 400;

#include "CImg.h"
#include <bits/stdtr1c++.h>
#include <unistd.h>
#include <sys/time.h>

using namespace std;
using namespace cimg_library;

const unsigned char red[] = { 255,0,0 }, green[] = { 0,255,0 }, blue[] = { 0,0,255 }, yellow[] = {255, 255, 0}, white[] = {255, 255, 255}, black[] = {0, 0, 0};

typedef vector <int> VI;
typedef vector <VI> VVI;
typedef long long LL;
typedef vector <LL> VLL;
typedef vector <double> VD;
typedef vector <VD> VVD;
typedef vector <string> VS;
typedef vector <VS> VVS;
typedef pair<int,int> PII;
typedef vector <PII> VPII;
typedef istringstream ISS;
typedef pair<double,double> PDD;

#define ALL(x) x.begin(),x.end()
#define REP(i,n) for (int i=0; i<(n); ++i)
#define FOR(var,pocz,koniec) for (int var=(pocz); var<=(koniec); ++var)
#define FORD(var,pocz,koniec) for (int var=(pocz); var>=(koniec); --var)
#define FOREACH(it, X) for(__typeof((X).begin()) it = (X).begin(); it != (X).end(); ++it)
#define PB push_back
#define PF push_front
#define MP(a,b) make_pair(a,b)
#define ST first
#define ND second
#define SIZE(x) (int)x.size()

template<class T> string i2s(T x) {ostringstream o; o << x; return o.str();}
template<class T1,class T2> ostream& operator<<(ostream &os, pair<T1,T2> &p) {os << "(" << p.first << "," << p.second << ")"; return os;}
template<class T> ostream& operator<<(ostream &os, vector<T> &v) {os << "{"; REP(i, (int)v.size()) {if (i) os << ", "; os << v[i];} os << "}"; return os;}
#define DB(a) {cerr << #a << ": " << (a) << endl; fflush(stderr); }

namespace Mytime{
  double start_time;
  static double last_call = 0;
   
  double get_time() {
    timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec+tv.tv_usec*1e-6;
  }

  void print_time(string s) {
    double x = get_time();
    fprintf(stderr,"%s cur=%.6lf lap=%.6lf\n",s.c_str(),x,x-last_call);
    last_call = x;
  }

  void init_time() {
    start_time = get_time();
    last_call = start_time;
  }
}

#define STOPER(name) Stoper name(#name);

struct Stoper;
vector<Stoper*> stoper_pointers;

struct Stoper {
  double used_time;
  double last_call;
  string name;
  void start() {
    last_call = Mytime::get_time();
  }
  void stop() {
    used_time += Mytime::get_time() - last_call;
  }

  Stoper(string s="") {
    used_time = 0.0;
    name=s;
    stoper_pointers.PB(this);
  }
}; 

STOPER(st_whole);
STOPER(st_inter);
STOPER(st_find);

/************************************************************************/
/************************ Code starts here ******************************/
/************************************************************************/

string TRAIN_DATA_DIR = "competition1/spacenet_TrainData/";
string TEST_DATA_DIR = "competition1/spacenet_TestData/";

string train_csv_path = TRAIN_DATA_DIR+"vectordata/summarydata/AOI_1_RIO_polygons_solution_3band.csv";
string train_images_path = TRAIN_DATA_DIR+"3band/3band_AOI_1_RIO_img";
string test_images_path = TEST_DATA_DIR+"3band/3band_AOI_2_RIO_img";
string train_labels_path = TRAIN_DATA_DIR+"labels/";
string train_heat_path = TRAIN_DATA_DIR+"heatmaps/";

const int TRAIN_IMAGES = 6940;
const int TEST_IMAGES = 2795;
const int IMAGE_WIDTH = 438;
const int IMAGE_HEIGHT = 406;
typedef vector<pair<double,double>> Polygon;
vector<Polygon> polygons[TRAIN_IMAGES * 2];
int seen[TRAIN_IMAGES * 2];
int strange_img[TRAIN_IMAGES];

Polygon translate(Polygon poly, double dx, double dy) {
  for (auto &p: poly) {
    p.ST += dx;
    p.ND += dy;
  }
  return poly;
}

double polygon_area(const Polygon &poly) {
  double res = 0.0;
  REP(i, SIZE(poly)-1) res += (poly[i+1].ST - poly[i].ST) * (poly[i].ND + poly[i+1].ND) / 2.0;
  return fabs(res);
}

VI positions(const char *txt, char c) {
  VI res;
  int i = 0;
  while (txt[i] != 10 && txt[i] != 13 && txt[i]) {
    if (txt[i] == c) res.PB(i);
    ++i;
  }
  return res;
}

Polygon parse_polygon(const char *txt, int &strange) {
  strange = 0;
  Polygon res;
  string s(txt);
  assert(s.substr(0, 10) == "POLYGON ((");
  VI commas = positions(txt, ',');
  commas.insert(commas.begin(), 9);
  int close = 0;
  while (txt[close] != ')') close++;
  static int calls = 0;
  calls++;
  for (auto i : commas) if (i < close) {
    double x, y;
    //if (calls < 10) DB(txt+i);
    x = atof(txt+i+1);
    int j = i+1;
    while (txt[j] != ' ') j++;
    y = atof(txt+j+1);
    j++;
    while (txt[j] != ' ') j++;
    j++;
    assert(txt[j] == '0');
    assert(txt[j+1] == ',' || txt[j+1] == ')');
    res.PB(MP(x,y));
  }
  assert(SIZE(res) > 3);
  if (calls < 2) {
    DB(txt);
    DB(res);
  }
  assert(s[SIZE(s)-1] == ')');
  assert(s[SIZE(s)-2] == ')');
  int j = 0;
  int a = 0, b = 0;
  while (txt[j]) {
    if (txt[j] == '(') a++;
    if (txt[j] == ')') b++;
    j++;
  }
  strange = a!=2;
  assert(a == b);
  /*
     TODO
  if (a != 2 || b != 2) {
    static int count = 0;
    count++;
    DB(a); DB(b);
    DB(count);
  }
  */
  //DB(txt);
  //DB(res);
  assert(fabs(res[0].ST - res.back().ST) < 1E-6);
  assert(fabs(res[0].ND - res.back().ND) < 1E-6);
  return res;
}

void mydraw_poly(CImg<unsigned char> &img, const Polygon &poly, const unsigned char *color, double opacity) {
  int w = img.width();
  int h = img.height();
  CImg<int> points(SIZE(poly),2);
  REP(k, SIZE(poly)){
    points(k,0) = (int)(poly[k].ST * w);
    points(k,1) = (int)(poly[k].ND * h);
  }
  img.draw_polygon(points, color, opacity);
  if (opacity < 0.01) {
    REP(k, SIZE(poly)-1) {
      pair<double,double> p1 = poly[k];
      pair<double,double> p2 = poly[k+1];
      img.draw_line((int)(p1.ST * w), (int)(p1.ND * h), (int)(p2.ST * w), (int)(p2.ND * h), color);
    }
  }
}

#define POINTT double
#define POINTR double 
struct POINT {
  POINTT x,y;
  POINT(POINTT wx, POINTT wy) : x(wx), y(wy) {}
  POINT(PDD p) {x = p.ST; y = p.ND;}
};
#define Det(p1,p2,w) (POINTR(p2.x-p1.x)*POINTR(w.y-p1.y)-POINTR(p2.y-p1.y)*POINTR(w.x-p1.x))
int sgn(double x){ return x > 0 ? 1 : (x < 0 ? -1 : 0); }

void pause(CImgDisplay &disp) {
  while (!disp.is_closed()) {
    disp.wait();
    if (disp.button() && disp.mouse_y()>=0) {
      //TODO
      const int y = disp.mouse_y();
      const int x = disp.mouse_x();
      DB(x);
      DB(y);
    }
    if (disp.is_keySPACE()) break;
  }
}

int pause_disp(CImgDisplay &main_disp) {
  while (!main_disp.is_closed()) {
    main_disp.wait();
    if (main_disp.is_keySPACE()) break;
    if (main_disp.is_keyARROWLEFT()) return -2;
    if (main_disp.is_keyARROWRIGHT()) return 0;
  }
  return 0;
}

int extract_id(string s) {
  int j = SIZE(s)-1;
  while (!isdigit(s[j])) j--;
  int p = 1;
  int i = 0;
  while (isdigit(s[j])) {
    i += p * (s[j] - '0');
    p *= 10;
    j--;
  }
  return i;
}

void extract_offsets(string s, double &scale, double &ox, double &oy) {
  int n = SIZE(s);
  scale = 1.0;
  ox = oy = 0.0;
  if (s[n-3] != '_') return;
  scale = 400.0 / (400.0 - (s[n-6] - 'a') * 20);
  oy = (s[n-4]-'a') / 400.0;
  ox = (s[n-2]-'a') / 400.0;
  fprintf(stderr, "offsets for %s are %.6lf %.6lf %.6lf\n", s.c_str(), scale, ox, oy);
}

const int HEAT_SIZE = 1000;
int heat_marked[HEAT_SIZE][HEAT_SIZE];
double heat_marked_dist[HEAT_SIZE][HEAT_SIZE];

inline double dist_from_segment(double x1, double y1, double x2, double y2, double x, double y) {
  x2 -= x1;
  y2 -= y1;
  x -= x1;
  y -= y1;
  double dot = x * x2 + y * y2;
  double d = hypot(x2, y2);
  double a = dot / d / d;
  if (a < 0 || a > 1) return 2.0;
  x -= a * x2;
  y -= a * y2;
  return hypot(x,y);
}

double dist_from_poly(const Polygon &poly, double x, double y) {
  double res = 2.0;
  for (auto p : poly) res = min(res, hypot(p.ST - x, p.ND - y));
  REP(i, SIZE(poly)-1) res = min(res, dist_from_segment(poly[i].ST, poly[i].ND, poly[i+1].ST, poly[i+1].ND, x, y));
  return res;
}

VVS load_csv_to_vvs(const char *filename) {
  const int T = 1000000;
  char txt[T];
  FILE *f = fopen(filename, "r");
  int lines = 0;
  VVS res;

  while (fgets(txt, T-1, f) != NULL) {
    ++lines;
    VI comma_pos = positions(txt, ',');
    comma_pos.PB(strlen(txt)-1);
    int last = -1;
    VS v;
    for (auto p : comma_pos) {
      v.PB(string(txt+last+1, txt+p));
      last = p;
    }
    if (lines <= 5) DB(v);
    res.PB(v);
  }
  fclose(f);
  fprintf(stderr, "read %d lines from file %s\n", lines, filename);
  return res;
}

struct Scan {
  string patient;
  int id; //scan id
  double x0, y0;
  double dx, dy;
  double slice_thickness;

  PDD mm_to_pixel(double px, double py) {
    double x = (px - x0) / dx;
    double y = (py - y0) / dy;
    x /= 512.0;
    y /= 512.0;
    return MP(x, y);
  }

  PDD pixel_to_mm(double px, double py) {
    px *= 512;
    py *= 512;
    double x = x0 + dx * px;
    double y = y0 + dy * py;
    return MP(x, y);
  }
};

vector<Scan> scans;

void load_scans(const char *filename) {
  VVS v = load_csv_to_vvs(filename);
  int n = SIZE(v);
  REP(i,n){
    assert(SIZE(v[i]) == 7);
    Scan s;
    s.patient = v[i][0];
    s.id = atoi(v[i][1].c_str());
    s.x0 = atof(v[i][2].c_str()); s.y0 = atof(v[i][3].c_str());
    s.dx = atof(v[i][4].c_str()); s.dy = atof(v[i][5].c_str());
    s.slice_thickness = atof(v[i][6].c_str());
    scans.PB(s);
  }
}

map<string, VS> dict_map;

void add_to_dict(string key, string res) {
  VI v = positions(res.c_str(), '|');
  v.PB(SIZE(res));
  VS pom;
  int last = -1;
  for (auto x : v) {
    pom.PB(res.substr(last+1, x-last-1));
    last = x;
  }
  DB(pom);
  assert(dict_map.count(key) == 0);
  dict_map[key] = pom;
}

VS parts;

void init_dictionary() {
  FILE *f = fopen("structure_dict.dat", "r");
  char txt[10000];
  while (fgets(txt, 10000, f)) {
    int pos = strlen(txt);
    while (txt[pos-1] == 10 || txt[pos-1] == 13) txt[--pos] = 0;
    int a = 0;
    while (txt[a] != '|') a++;
    string part(txt, txt+a);
    DB(part);
    parts.PB(part);
    DB(txt);
    add_to_dict(part,txt);
  }
  DB(parts);
  fclose(f);
}

//returns 0 in case of failure
int get_dat_id(string patient, int scan_id, string what) {
  string filename = DATA_PATH+"/"+patient+"/structures.dat";
  DB(filename);
  FILE *f = fopen(filename.c_str(), "r");
  const int T = 10000;
  char txt[T];
  fgets(txt, T-1, f);
  //fscanf(f, "%s", txt);
  VI v = positions(txt, '|');
  v.PB(strlen(txt)-1);
  DB(txt);
  int last = -1;
  int found = -1;
  int i = 0;
  for (auto x : v) {
    string s(txt+last+1, txt+x);
    int ok = 0;
    for (auto &ss : dict_map[what]) if (ss == s) {
      ok = 1;
      break;
    }
    if (ok) {
      found = i;
      break;
    }
    last = x;
    ++i;
  }
  return found+1;
}

/*
// Convert from image space (pixels) to physical space (millimeters)
private P2 pixelToMm(P2 p, Slice slice) {
  double x = (p.x * slice.dx) + slice.x0;
  double y = (p.y * slice.dy) + slice.y0;
  return new P2(x, y);
}
*/

// Convert from physical space (millimeters) to image space (pixels)
/*
private P2 mmToPixel(P2 p, Slice slice) {
  double x = (p.x - slice.x0) / slice.dx;
  double y = (p.y - slice.y0) / slice.dy;
  return new P2(x, y);
}
*/

void generate_heatmaps(int heat_size) {
  char txt[1000000];
  CImg<unsigned char> image, heat;
  CImgDisplay disp_img, disp_heat;
  FILE *f = fopen(".masks", "w");
  for (auto &s : scans) {
    VI mask;
    REP(i, SIZE(parts)) {
      string part = parts[i];
      DB(s.patient + " " + i2s(s.id));
      CImg<unsigned char> img((DATA_PATH+"/"+s.patient+"/pngs/"+i2s(s.id)+".png").c_str());
      DB("here");
      CImg<unsigned char> heat(heat_size, heat_size, 1, 1, 0);
      int dat_id = get_dat_id(s.patient, s.id, part);
      mask.PB(dat_id > 0);
      if (dat_id > 0) {
        string filename = DATA_PATH + "/" + s.patient + "/contours/" + i2s(s.id) + "." + i2s(dat_id) + ".dat";
        DB(filename);
        FILE *f = fopen(filename.c_str(), "r");
        while (f && fscanf(f, "%s", txt) == 1) {
          VI v = positions(txt, ',');
          v.PB(strlen(txt));
          assert(SIZE(v) % 3 == 0);
          int i = 0;
          Polygon poly;
          int last = -1;
          while (i < SIZE(v)) {
            double x = atof(string(txt+last+1, txt+v[i]).c_str());
            double y = atof(string(txt+v[i]+1, txt+v[i+1]).c_str());

            poly.PB(s.mm_to_pixel(x,y));
            last = v[i+2];
            i += 3;
          }
          //DB(poly);
          mydraw_poly(img, poly, red, 0.5);
          mydraw_poly(img, poly, red, 1.0);
          mydraw_poly(img, poly, red, 0);
          mydraw_poly(heat, poly, white, 1.0);
        }
      }
      heat.save(("heatmaps/"+s.patient+"_"+i2s(s.id)+"_"+i2s(i)+".jpg").c_str());

      //  disp_img = img;
      //  disp_heat = heat;
      //  pause(disp_img);
    }
    fprintf(f, "%s,%d", s.patient.c_str(), s.id);
    for (auto x : mask) fprintf(f, ",%d", x);
    fprintf(f, "\n");
  }
  fclose(f);
}

int processed[MAX_PID][MAX_SID];
int stats_yes[MAX_PID][256];
int stats_no[MAX_PID][256];
double scores[MAX_PID][256];

void compute_score(int pid) {
  int which = 0;
  string ss = i2s(pid);
  while (SIZE(ss) < 3) ss = "0" + ss;
  while (which < SIZE(scans) && scans[which].patient != "ANON_LUNG_TC"+ss) which++;
  assert(which < SIZE(scans));

  FOR(threshold,0,255) {
    double tp = 0;
    double fn = 0;
    double fp = 0;
    FOR(i,threshold,255) fp += stats_no[pid][i];
    REP(i,256) if (i < threshold) fn += stats_yes[pid][i];
    else tp += stats_yes[pid][i];

    double mult = scans[which].dx * scans[which].dy * scans[which].slice_thickness;
    tp *= mult;
    fn *= mult;
    fp *= mult;

    double t = tp + fn;
    double fn1 = (double)fn * t / max(0.01, (double)tp);
    double e = fn1 + fp;
    double a = pow(36.0 * 2.0 * acos(0.0) * t * t, 1.0 / 3);
    double r1 = e / t;
    double l = 10;
    double r2 = e / (l * a);
    double rr = (r1+r2) / 2.0;
    double score = exp(-rr);
    scores[pid][threshold] = score;
    fprintf(stderr, "pid=%d threshold=%d tp=%.6lf fp=%.6lf fn=%.6lf r1=%.6lf r2=%.6lf rr=%.6lf score=%.6lf\n", pid, threshold, tp, fp, fn, r1, r2, rr, score);
  }
}

const int K = 512;
int turn = 0;
int board[K][K];
int vis[K][K];
int total_polys;
int dx[] = {0, 1, 0, -1};
int dy[] = {1, 0, -1, 0};

VPII kraw[K+5][K+5];

inline int between(int a,int b, int c) {
  return a >= b && a <= c;
}

Polygon bfs(int i, int j) {
  vis[i][j] = 1;

  VPII q;
  q.PB(MP(i,j));
  REP(foo, SIZE(q)) {
    i = q[foo].ST;
    j = q[foo].ND;
    REP(r, 4) {
      int ni = i + dx[r];
      int nj = j + dy[r];
      if (between(ni, 0, K-1) && between(nj, 0, K-1) && board[ni][nj] == turn && !vis[ni][nj]) {
        vis[ni][nj] = 1;
        q.PB(MP(ni,nj));
      }
    }
  }

  for (auto p: q) {
    i = p.ST;
    j = p.ND;
    REP(di,2) REP(dj,2) kraw[i+di][j+dj].clear();
  }

  for (auto p : q) {
    i = p.ST;
    j = p.ND;
    REP(r, 4) {
      int ni = i + dx[r];
      int nj = j + dy[r];
      if (between(ni, 0, K-1) && between(nj, 0, K-1) && board[ni][nj] != turn) {
        if (r == 0) {
          kraw[i][j+1].PB(MP(i+1,j+1));
        } else if (r == 1) {
          kraw[i+1][j+1].PB(MP(i+1,j));
        } else if (r == 2) {
          kraw[i+1][j].PB(MP(i,j));
        } else { // r == 3
          kraw[i][j].PB(MP(i,j+1));
        }
      }
    }
  }

  sort(ALL(q));
  i = q[0].ST;
  j = q[0].ND;
  assert(SIZE(kraw[i][j]) == 1);
  Polygon poly;
  int starti = i, startj = j;
  int r = -1;
  do {
    poly.PB(MP(i / (double)K, j / (double)K));
    if (SIZE(kraw[i][j]) == 1) {
      PII p = kraw[i][j][0];
      REP(rr, 4) if (p.ST == i + dx[rr] && p.ND == j + dy[rr]) r = rr;
      assert(r >= 0);
      i = p.ST;
      j = p.ND;
    } else if (SIZE(kraw[i][j]) == 2) {
      r = (r+1) % 4;
      int ni = i + dx[r];
      int nj = j + dy[r];
      assert(kraw[i][j][0] == MP(ni,nj) || kraw[i][j][1] == MP(ni,nj));
      i = ni;
      j = nj;
    } else assert(0);
  } while (i != starti || j != startj);
  poly.PB(MP(i / (double)K, j / (double)K));
  double area = polygon_area(poly) * K * K;
  DB(area);
  return poly;
}

void find_polygons(const VPII &marked, CImg<unsigned char> &img, FILE *f, string pref, Scan &scan) {
  ++turn;
  for (auto p : marked) board[p.ST][p.ND] = turn, vis[p.ST][p.ND] = 0;

  for (auto p : marked) if (!vis[p.ST][p.ND]) {
    Polygon poly = bfs(p.ST, p.ND);
    mydraw_poly(img, poly, red, 0.0);
    mydraw_poly(img, poly, red, 0.5);
    DB(poly);
    ++total_polys;
    DB(total_polys);
    fprintf(f, "%s", pref.c_str());
    for (auto p : poly) {
      PDD x = scan.pixel_to_mm(p.ST, p.ND);
      fprintf(f, ",%.2lf,%.2lf", x.ST, x.ND);
    }
    fprintf(f, "\n");
  }
}

string pad3(int x) {
  char txt[10];
  sprintf(txt, "%03d", x);
  return txt;
}

double get_thickness(int pid) {
  DB(pid);
  int which = 0;
  double mult = scans[which].dx * scans[which].dy * scans[which].slice_thickness;
  string ss = i2s(pid);
  while (SIZE(ss) < 3) ss = "0" + ss;
  while (which < SIZE(scans) && scans[which].patient != "ANON_LUNG_TC"+ss) which++;
  return scans[which].dx * scans[which].dy * scans[which].slice_thickness;
}

double num_pix[MAX_PID+10][256];

void find_pixels(const vector<pair<PII,string>> &v) {
  REP(foo, SIZE(v)) {
    int sid = v[foo].ST.ND;
    int pid = v[foo].ST.ST;
    string s = v[foo].ND;
    CImg<unsigned char> pred(s.c_str());
    double mult = get_thickness(pid);
    REP(x,512) REP(y,512) num_pix[pid][(unsigned int) pred(x,y,0,0)] += mult;
  }
}

int compute_threshold(int pid, int threshold) {
  double pixels = 0.0;
  int tt = threshold;
  FOR(i,tt,255) pixels += num_pix[pid][i];
  while (tt > 1 && pixels < 4000.0) {
    //TODO
    pixels += num_pix[pid][--tt];
  }
  return tt;
}

void process_predictions(string filename, int threshold) {
  char txt[1000000];
  DB(filename);
  fstream in(filename);
  string s;
#ifdef VIS
  CImgDisplay disp_img, disp_truth, disp_pred, disp_rounded;
#endif
  vector<pair<PII,string>> v;
  set<int> set_pid;
  while (in >> s) {
    VI sep = positions(s.c_str(), '_');
    int a = sep[SIZE(sep)-2];
    int b = sep[SIZE(sep)-1];
    int sid = atoi(s.substr(a+1,b-a).c_str());
    int p = atoi(s.substr(sep[SIZE(sep)-3]+3).c_str());
    //DB(s);
    //DB(p);
    //DB(sid);
    v.PB(MP(MP(p,sid),s));
    set_pid.insert(p);
  }
  sort(ALL(v));
  int last = -1;
  FILE *f = fopen("submission.csv", "w");
  find_pixels(v);
  REP(foo, SIZE(v)) {
    int sid = v[foo].ST.ND;
    int pid = v[foo].ST.ST;
    assert(sid >= 0 && sid < MAX_SID);
    assert(pid >= 0 && pid < MAX_PID);


    if (pid != last) {
      if (last >= 0) compute_score(last);
      last = pid;
    }
    string s = v[foo].ND;
    //int pos = 0;
    //while (s[pos] != '_') pos++;
    //string pref = s.substr(0,pos+1);
    //s = s.substr(pos+1);
    DB(s);
    //CImg<unsigned char> img(("example_extracted/"+s.patient+"/pngs/"+i2s(s.id)+".png").c_str());
    VI sep = positions(s.c_str(), '_');
    string spatient = "ANON_LUNG_TC"+pad3(pid);
    DB(spatient);
    DB(sid);

    int which = -1;
    REP(i, SIZE(scans)) if (scans[i].patient == spatient && scans[i].id == sid) which = i;
    assert(which >= 0);


    CImg<unsigned char> rounded(512, 512, 1, 3, 0);
    VPII marked;

    CImg<unsigned char> pred(s.c_str());
    int cnt_marked = 0;
    if (VISUALIZE) {
      CImg<unsigned char> truth(("unpacked_heatmaps/"+spatient+"_"+i2s(sid)+"_0.jpg").c_str());
#ifdef VIS
      disp_pred.set_title(s.c_str());
      disp_truth.set_title(s.c_str());
#endif
      VI yes(256, 0);
      VI no(256, 0);
      REP(x,512) REP(y,512) {
        unsigned char t = truth(x,y,0,0);
        if (t >= 126) {
          yes[(unsigned int) pred(x,y,0,0)]++;
        } else {
          no[(unsigned int) pred(x,y,0,0)]++;
        } 
        /*else {
          rounded(x,y,0,0) = 0;
          rounded(x,y,0,1) = 0;
          rounded(x,y,0,2) = 0;
        }
        */
      }
      if (!processed[pid][sid]) {
        processed[pid][sid] = 1;
        REP(i, 256) stats_yes[pid][i] += yes[i];
        REP(i, 256) stats_no[pid][i] += no[i];
      }

      int tt = compute_threshold(pid, threshold);
      REP(x,512) REP(y,512) {
        if (pred(x,y,0,0) >= tt) {
          rounded(x,y,0,0) = 255;
          rounded(x,y,0,1) = 255;
          rounded(x,y,0,2) = 255;
          marked.PB(MP(x,y));
          cnt_marked++;
        } 
      }
      DB(yes);
      DB(no);

      /*
      int dat_id = get_dat_id(spatient, sid, "gtv");
      string filename = "example_extracted/" + spatient + "/contours/" + i2s(sid) + "." + i2s(dat_id) + ".dat";
      FILE *f = fopen(filename.c_str(), "r");
      while (f && fscanf(f, "%s", txt) == 1) {
        VI v = positions(txt, ',');
        v.PB(strlen(txt));
        assert(SIZE(v) % 3 == 0);
        int i = 0;
        Polygon poly;
        int last = -1;

        int which = -1;
        REP(i, SIZE(scans)) if (scans[i].patient == spatient && scans[i].id == sid) which = i;
        assert(which >= 0);

        while (i < SIZE(v)) {
          double x = atof(string(txt+last+1, txt+v[i]).c_str());
          double y = atof(string(txt+v[i]+1, txt+v[i+1]).c_str());

          poly.PB(scans[which].mm_to_pixel(x,y));
          last = v[i+2];
          i += 3;
        }
        DB(poly);
        mydraw_poly(pred, poly, red, 0.5);
        mydraw_poly(truth, poly, red, 1.0);
      }
      */
      double mult = scans[which].dx * scans[which].dy * scans[which].slice_thickness;
      DB(cnt_marked);
      DB(cnt_marked * mult);

      find_polygons(marked, rounded, f, spatient+","+i2s(sid), scans[which]);
#ifdef VIS
      disp_truth = truth;
      disp_pred = pred;
      disp_rounded = rounded;
      foo += pause_disp(disp_rounded);
#endif
    } else {
      int tt = compute_threshold(pid, threshold);
      REP(x,512) REP(y,512) {
        if (pred(x,y,0,0) >= tt) {
          rounded(x,y,0,0) = 255;
          rounded(x,y,0,1) = 255;
          rounded(x,y,0,2) = 255;
          marked.PB(MP(x,y));
          cnt_marked++;
        }
      }
      find_polygons(marked, rounded, f, spatient+","+i2s(sid), scans[which]);
#ifdef VIS
      disp_rounded = rounded;
      disp_pred = pred;
      foo += pause_disp(disp_rounded);
#endif
    }
  }
  fclose(f);
  int cnt = 0;
  VD sum_scores(256, 0.0);
  double mm = 0.0;
  for (auto pid : set_pid) {
    cnt++;
    double m = 0.0;
    REP(i, 256) {
      sum_scores[i] += scores[pid][i];
      m = max(m, scores[pid][i]);
    }
    mm += m;
  }

  DB(cnt);
  DB(mm / SIZE(set_pid));
  REP(i,256) fprintf(stderr, "ave score for threshold %d is %.6lf\n", i, sum_scores[i]/cnt);

  for (double l = 400.0; l < 40001; l *= 1.01) {
    for (int t = 109; t <= 109; t++) {
      mm = 0.0;
      double pixels = 0.0;
      for (auto pid : set_pid) {
        double thick = get_thickness(pid);

        int i = 255;
        while (i > 0 && (i >= t || pixels < l)) {
          --i;
          pixels += stats_yes[pid][i] * thick;
          pixels += stats_no[pid][i] * thick;
        }
        mm += scores[pid][i];
      }
      fprintf(stderr, "l %.6lf t %d ave %.6lf\n", l, t, mm / SIZE(set_pid));
    }
  }
}

int main(int argc, char **argv) {
  /*
  CImg<unsigned char> img("example_extracted/ANON_LUNG_TC010/pngs/160.png");
  CImgDisplay disp;
  disp = img;
  pause(disp);
  */
  int proc_predictions = 0;
  string pred_filename;
  int threshold;
  int heat = 0;
  int heat_size;
  for (int i = 1; i < argc; ++i) {
    if (string(argv[i]) == "--heat") {
      heat = 1;
      ++i;
      heat_size = atoi(argv[i]);
    } else if (string(argv[i]) == "--vis") {
      VISUALIZE = 1;
    } else if (string(argv[i]) == "--predictions") {
      ++i;
      pred_filename = argv[i];
      proc_predictions = 1;
      ++i;
      threshold = atoi(argv[i]);
    } else if (string(argv[i]) == "--path") {
      DATA_PATH = argv[++i];
    } else assert(0);
  } 
  DB(DATA_PATH);
  DB(proc_predictions);
  DB(heat);
  init_dictionary();

  if (heat) {
    load_scans("scans.csv");
  } else {
    load_scans("scans_all.csv");
    load_scans("scans_test.csv");
  }

  if (proc_predictions) {
    process_predictions(pred_filename, threshold);
  } else {
    generate_heatmaps(heat_size);
  }
  return 0;
}

