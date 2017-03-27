#include "location.h" 
#include <bits/stdc++.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <dirent.h>
#define SSTR( x ) dynamic_cast< std::ostringstream & >( ( std::ostringstream() << std::dec << x ) ).str()
#define SSTRF( x ) dynamic_cast< std::ostringstream & >( ( std::ostringstream() << fixed << setprecision(2) << x ) ).str()
#define RGBINT( x ) ( (x.r << 16) + (x.g << 8) + x.b )
#define GCOST(x) ( sqrt((x)*(1 - (x)) ) )

#include "image.h"
#include "misc.h"
#include "pnmfile.h"
#include "segment-image.h"
#include "psimpl.h"

#define OOB 1

using namespace std;

const int DARK = 55;
const int W = 512;
const int H = 512;
const double ATLEAST = 0.5;
const double QEXPAND = 0.15;
const bool EXPAND = true;

//double THRESHOLD = 0.07; //0.06; //0.03; //0.1; //0.05;//0.117; //0.132; //0.2; //0.125;
vector <double> THRESHOLD = {0.02, 0.04, 0.06, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.145, 0.15, 0.155, 0.16, 0.165, 0.17, 0.18, 0.19, 0.2, 0.25};

bool USEMSE = false;
const int CLUSTERS = 2;
int FEATURETRY = 5;
const int FEATURES = 23 + 13 + 10 + 4 + 6; //26 + 5 + 5 + 1;
const int MAXLEVEL = 50;
int MINNODEMSE = 5;
int MINNODE = 5;
int TREES = 500;
int BORDER = 10000000;
const double EPS = 1e-10;
const int MAXSAMPLESIZE = 300000; //10000000;
vector <int> featureScore;
vector <int> featureScoreC;
const int SAMPLES = 4000000;
float FEAT[FEATURES * SAMPLES];
float RESULT[SAMPLES];
const double PI = 3.141592653589793238;

double getTime() {
	unsigned long long timelo, timehi;
    __asm__ volatile ("rdtsc" : "=a" (timelo), "=d" (timehi));
    return ((timehi << 32) + timelo) / 2.5e9;
}
// random generator
unsigned long long nowRand = 1;
void seedBig(unsigned long long seed){
	nowRand = seed;
}
unsigned long long randBig(){
	nowRand = ((nowRand * 6364136223846793005ULL + 1442695040888963407ULL) >> 1);
	return nowRand;
}
string int2len(int v,int l){
	string ret=SSTR(v);
	int dig=floor(log10(v+0.1))+1;
	while(ret.length()<l) ret=" "+ret;
	return ret;
}
string string2len(string ret,int l){
	while(ret.length()<l) ret+=" ";
	return ret;
}
vector <int> selectFeatures(){
	set <int> temp;
	while(temp.size()!=FEATURETRY){
		temp.insert(randBig()%FEATURES);
	}
	vector<int> result(temp.begin(),temp.end());
	//random_shuffle(result.begin(),result.end());
	return result;
}
class Item{
public:
	vector <float> feature;
	float result;
	#if OOB==1
	int oobCount;
	float oobResult;
	#endif
	Item(){
		#if OOB==1
		oobCount = 0;
		oobResult = 0;
		#endif
	}
	Item(int id, const vector <float>& feat){
		#if OOB==1
		oobCount = 0;
		oobResult = 0;
		#endif
		if(feat.size() > FEATURES){
			RESULT[id] = feat.back();
			for(int i = 0; i < FEATURES; i++) FEAT[i*SAMPLES + id] = feat[i];
			result = feat.back();
		}
		else{
			feature = feat;
			result = 0;
		}
	}
};
vector <Item> trainData;
int SORTBY = 0;
int SORTBYSAMPLES = 0;
class Node{
  public:
	int left;
	int right;
	int feature;
	double value;
	int level;
	int counts[CLUSTERS];
	int total;
	double sumX;
	Node(){
		left = -1;
		right = -1;
		feature = -1;
		value = 0;
		level = -1;
		for(int i = 0; i < CLUSTERS; i++) counts[i] = 0;
		total = 0;
	}
	Node(int lev, const vector <int>& cou, int tot){
		left = -1;
		right = -1;
		feature = -1;
		value = 0;
		level = lev;
		copy(cou.begin(), cou.end(), counts);
		total = tot;
	}
	Node(int lev, int tot, const double& sumx){
		left = -1;
		right = -1;
		feature = -1;
		value = 0;
		level = lev;
		total = tot;
		sumX = sumx;
	}
};
class Tree{
  public:
	vector <Node> node;
	double resultAtNode(int nodeId, const Item& ad) const{
		if(node[nodeId].left == -1){
			return node[nodeId].sumX / node[nodeId].total;
		}
		if(ad.feature[node[nodeId].feature] <= node[nodeId].value) return resultAtNode(node[nodeId].left, ad);
		return resultAtNode(node[nodeId].right, ad);
	}
	double assignResult(const Item& ad) const{
		return resultAtNode(0, ad);
	}
	double OOBresultAtNode(int nodeId, int i) const{
		if(node[nodeId].left == -1){
			return node[nodeId].sumX / node[nodeId].total;
		}
		if(FEAT[node[nodeId].feature*SAMPLES + i] <= node[nodeId].value) return OOBresultAtNode(node[nodeId].left, i);
		return OOBresultAtNode(node[nodeId].right, i);
	}
	double assignOOBResult(int i) const{
		return OOBresultAtNode(0, i);
	}
	double ratioAtNode(int nodeId, const Item& ad) const{
		if(node[nodeId].left == -1){
			double sum = 0;
			for(int cluster = 0; cluster < CLUSTERS; cluster++){
				sum += node[nodeId].counts[cluster] * cluster;
			}
			return sum / node[nodeId].total;
		}
		if(ad.feature[node[nodeId].feature] <= node[nodeId].value) return ratioAtNode(node[nodeId].left, ad);
		return ratioAtNode(node[nodeId].right, ad);
	}
	double assignRatio(const Item& ad) const{
		return ratioAtNode(0, ad);
	}
	double OOBratioAtNode(int nodeId, int id) const{
		if(node[nodeId].left == -1){
			double sum = 0;
			for(int cluster = 0; cluster < CLUSTERS; cluster++){
				sum += node[nodeId].counts[cluster] * cluster;
			}
			return sum / node[nodeId].total;
		}
		if(FEAT[node[nodeId].feature*SAMPLES + id] <= node[nodeId].value) return OOBratioAtNode(node[nodeId].left, id);
		return OOBratioAtNode(node[nodeId].right, id);
	}
	double assignOOBRatio(int id) const{
		return OOBratioAtNode(0, id);
	}
	void divideNode(int nodeIndex, vector <int>& sample){
		int n = sample.size();
		int nonzero = 0;
		for(int i = 0; i < CLUSTERS; i++){
			if(node[nodeIndex].counts[i] > 0) nonzero++;
			if(nonzero > 1) break;
		}
		if(node[nodeIndex].level < MAXLEVEL - 1 && nonzero > 1 && node[nodeIndex].total > MINNODE){
	    	vector <int> feaID = selectFeatures();
	    	double minCost = 1e30;
	    	int bestF = -1;
	    	double bestValue = 0;
	    	int bestI = 0;
	    	vector <int> bestC1(CLUSTERS, 0);
	    	int bestTotalL = 0;
			for(int f = 0; f < FEATURETRY; f++){
				SORTBY = feaID[f];
				SORTBYSAMPLES = SORTBY * SAMPLES;
				sort(sample.begin(), sample.end(), [&](int aa, int bb){return FEAT[SORTBYSAMPLES + aa] < FEAT[SORTBYSAMPLES + bb];});
				
				/*int FSIZ=FSIZE[fi*FEATURES+SORTBY];
				int bucket[FSIZ]={};
				for(int j=0;j<n;j++){
					bucket[FEAT[SORTBYSAMPLES+sample[j]]]++;
				}
				int cumSum[FSIZ+1];
				cumSum[0]=0;
				for(int i=0;i<FSIZ;i++) cumSum[i+1]=cumSum[i]+(bucket[i]--);
				vector <int> temp=sample;
				for(int i=0;i<n;i++){
					int v=FEAT[SORTBYSAMPLES+temp[i]];
					sample[cumSum[v]+(bucket[v]--)]=temp[i];
				}*/
				
				vector <int> c1(CLUSTERS, 0);
	    		int totalL = 0;
	    		for(int i = 0; i < n-1; i++){
	    			c1[int(RESULT[sample[i]])]++;
	    			totalL++;
	    			if(FEAT[SORTBYSAMPLES + sample[i+1]] > FEAT[SORTBYSAMPLES + sample[i]]){
	    			    double costL = 0.0;
						double costR = 0.0;
						for(int cl = 0; cl < CLUSTERS; cl++){
							costL += GCOST(c1[cl] / static_cast<double>(totalL));
							costR += GCOST((node[nodeIndex].counts[cl] - c1[cl]) / static_cast<double>(n - totalL));
						}
						double cost = (totalL * costL + (n - totalL) * costR) / n;
						if(cost < minCost && i >= n/BORDER && i < n - n/BORDER){
	    			    	minCost = cost;
	    			    	bestF = feaID[f];
	    			    	bestValue = FEAT[SORTBYSAMPLES + sample[i]];
							bestI = i;
	    			    	bestC1 = c1;
	    			    	bestTotalL = totalL;
	    			    }
					}
	    		}
	    	}
	    	if(bestF >= 0){
				//if(bestTotalL == node[nodeIndex].total) cerr << node[nodeIndex].total << " " << n << endl;
				featureScore[bestF] += n;
				featureScoreC[bestF]++;
		    	vector <int> sampleLeft; sampleLeft.reserve(bestI + 1);
		    	vector <int> sampleRight; sampleRight.reserve(n - bestI - 1);
		    	SORTBYSAMPLES = bestF * SAMPLES;
				for(int i = 0; i < n; i++){
		    		if(FEAT[SORTBYSAMPLES + sample[i]] <= bestValue){
						sampleLeft.push_back(sample[i]);
					}
		    		else sampleRight.push_back(sample[i]);
		    	}
		    	/*if(sampleLeft.size() != bestTotalL){
					cout << "!" << sampleLeft.size() << " " << bestTotalL << endl;
					cout << bestValue << " " << bestI << endl;
					for(int i = 0; i < n; i++){
						cout << FEAT[SORTBYSAMPLES + sample[i]] << endl;
			    	}
			    	cout << " ------------------------------------- " << endl;
				}*/
		        node[nodeIndex].feature = bestF;
		    	node[nodeIndex].value = bestValue;
		    	node.push_back(Node(node[nodeIndex].level + 1, bestC1, bestTotalL));
		    	//if(bestTotalL <= 0) cerr << "L" << bestTotalL << endl;
		    	node[nodeIndex].left = node.size() - 1;
		    	vector <int> c2(CLUSTERS, 0);
		    	for(int i = 0; i < CLUSTERS; i++){
		    		c2[i] = node[nodeIndex].counts[i] - bestC1[i];	
		    	}
		    	node.push_back(Node(node[nodeIndex].level + 1, c2, node[nodeIndex].total - bestTotalL));
		    	//if(node[nodeIndex].total - bestTotalL <= 0) cerr << "R" << node[nodeIndex].total - bestTotalL << endl;
		    	
		    	node[nodeIndex].right = node.size() - 1;
			    divideNode(node[nodeIndex].left, sampleLeft);
				divideNode(node[nodeIndex].right, sampleRight);
			}
		}
	}
	void divideNodeMSE(int nodeIndex, vector <int>& sample){
		int n = sample.size();
		if(node[nodeIndex].level < MAXLEVEL-1 && node[nodeIndex].total > MINNODEMSE){
			vector <int> feaID = selectFeatures();
			double minCost = 1e30;
			int bestF = -1;
			double bestValue = 0;
			int bestI = 0;
			double bestSumXL = 0;
			int bestTotalL = 0;
			for(int f = 0; f < FEATURETRY; f++){
				SORTBY = feaID[f];
				SORTBYSAMPLES = SORTBY*SAMPLES;
				sort(sample.begin(), sample.end(), [&](int aa, int bb){return FEAT[SORTBYSAMPLES+aa] < FEAT[SORTBYSAMPLES+bb];});
				
				//sortFast(sample);
				
				/*int bucket[FSIZE[SORTBY]]={};
				for(int j=0;j<n;j++){
					bucket[FEAT[SORTBYSAMPLES+sample[j]]]++;
				}
				int cumSum[FSIZE[SORTBY]+1];
				cumSum[0]=0;
				for(int i=0;i<FSIZE[SORTBY];i++) cumSum[i+1]=cumSum[i]+(bucket[i]--);
				vector <int> temp=sample;
				for(int i=0;i<n;i++){
					int v=FEAT[SORTBYSAMPLES+temp[i]];
					sample[cumSum[v]+(bucket[v]--)]=temp[i];
				}*/
				
				
				double sumXL = 0;
				int totalL = 0;
	    		for(int i = 0; i < n - 1; i++){
					sumXL += RESULT[sample[i]];
					totalL++;
					if(FEAT[SORTBYSAMPLES + sample[i+1]] > FEAT[SORTBYSAMPLES + sample[i]]){
						double cost = -sumXL*sumXL/totalL - (node[nodeIndex].sumX-sumXL) * (node[nodeIndex].sumX-sumXL) / (n-totalL);
						if(cost < minCost && i >= n/BORDER && i < n - n/BORDER){
	    			    	minCost = cost;
	    			    	bestF = SORTBY;
							bestValue = FEAT[SORTBYSAMPLES+sample[i]];
							bestI = i;
	    			    	bestSumXL = sumXL;
	    			    	bestTotalL = totalL;
	    			    }
					}
	    		}
	    	}
	    	if(bestF >= 0){
	    		featureScore[bestF] += n;
	    		featureScoreC[bestF]++;
				vector <int> sampleLeft; sampleLeft.reserve(bestI + 1);
		    	vector <int> sampleRight; sampleRight.reserve(n - bestI - 1);
				SORTBYSAMPLES = bestF*SAMPLES;
				for(int i = 0; i < n; i++){
					if(FEAT[SORTBYSAMPLES + sample[i]] <= bestValue){
						sampleLeft.push_back(sample[i]);
					}
		    		else sampleRight.push_back(sample[i]);
		    	}
		        node[nodeIndex].feature = bestF;
		    	node[nodeIndex].value = bestValue;
		    	node.push_back(Node(node[nodeIndex].level+1, bestTotalL, bestSumXL));
		    	node[nodeIndex].left = node.size() - 1;
		    	node.push_back(Node(node[nodeIndex].level+1, node[nodeIndex].total-bestTotalL, node[nodeIndex].sumX-bestSumXL));
		    	node[nodeIndex].right = node.size() - 1;
				
				divideNodeMSE(node[nodeIndex].left,sampleLeft);
				divideNodeMSE(node[nodeIndex].right,sampleRight);
			}
		}
	}
	Tree(){
	}
};
double forestAssignResult(const vector <Tree>& tree, const Item& item){
	double result = 0;
	for(int t = 0; t < tree.size(); t++){
		result += USEMSE ? tree[t].assignResult(item) : tree[t].assignRatio(item);
	}
	return result / tree.size();
}
Tree buildTree(){
	int n = trainData.size();
	int ns = min(n, MAXSAMPLESIZE);
	vector <int> sample;
	#if OOB==1
	vector <bool> isOOB(n, true);
	#endif
	sample.resize(ns);
	Tree tree;
	tree.node.resize(1, Node());
	tree.node[0].level = 0;
	for(int i = 0; i < ns; i++){
		sample[i] = randBig() % n;
		#if OOB==1
		isOOB[sample[i]] = false;
		#endif
		tree.node[0].counts[int(RESULT[sample[i]])]++;
	}
	tree.node[0].total = ns;
	tree.divideNode(0,sample);
	#if OOB==1
	for(int i = 0; i < n; i++) if(isOOB[i]){
		trainData[i].oobCount++;
		trainData[i].oobResult += tree.assignOOBRatio(i);
	}
	#endif
	return tree;
}
Tree buildTreeMSE(){
	int n = trainData.size();
	int ns = min(n, MAXSAMPLESIZE);
	vector <int> sample;
	#if OOB==1
	vector <bool> isOOB(n, true);
	#endif
	sample.resize(ns);
	Tree tree;
	tree.node.resize(1, Node());
	tree.node[0].level = 0;
	tree.node[0].sumX = 0;
	for(int i = 0; i < ns; i++){
		sample[i] = randBig() % n;
		#if OOB==1
		isOOB[sample[i]] = false;
		#endif
		tree.node[0].sumX += RESULT[sample[i]];
	}
	tree.node[0].total = ns;
	tree.divideNodeMSE(0, sample);
	#if OOB==1
	for(int i = 0; i < n; i++) if(isOOB[i]){
		trainData[i].oobCount++;
		trainData[i].oobResult += tree.assignOOBResult(i);
	}
	#endif
	return tree;
}

inline rgb INTRGB(int x){
	rgb ret;
	ret.r = x / 65536;
	ret.g = x / 256 % 256;
	ret.b = x % 256;
	return ret;
}
vector <string> splitBy(const string& text, char by){   // split string by by
	vector <string> vys;
	stringstream ss(text);
    string word;
    while(getline(ss,word,by)){
        vys.push_back(word);
    }
    return vys;
}
vector <string> listdir(const string& folder){
	DIR *dir;
	struct dirent *ent;
	vector <string> files;
	if((dir = opendir(folder.c_str())) != NULL){
		while((ent = readdir(dir)) != NULL){
			string temp = ent->d_name;
			if(temp != "." && temp != "..") files.push_back(temp);
		}
		closedir(dir);
	} else{
		cerr << "Could not open directory [" << folder << "]" << endl;
	}
	sort(files.begin(), files.end());
	return files;
}
vector <string> listdirdir(const string& folder){
	DIR *dir;
	struct dirent *ent;
	vector <string> files;
	struct stat s;
	if((dir = opendir(folder.c_str())) != NULL){
		while((ent = readdir(dir)) != NULL){
			stat((folder+ent->d_name).c_str(), &s);
			//if(S_ISDIR(s.st_mode)){
			//if((s.st_mode & S_IFMT) == S_IFDIR){
			if (s.st_mode & S_IFDIR){
			//if(ent->d_type == 0x4){
				string temp = ent->d_name;
				if(temp != "." && temp != "..") files.push_back(temp);
			}
		}
		closedir(dir);
	} else{
		cerr << "Could not open directory [" << folder << "]" << endl;
	}
	sort(files.begin(), files.end());
	return files;
}


vector <int> simplifyV(const vector <int>& path, int WW, int vertices){
	vector <double> temp(path.size()*2);
	for(int i = 0; i < path.size(); i++){
		temp[2 * i] = path[i] % WW;
		temp[2 * i + 1] = path[i] / WW;
	}
	vector <double> temp2;
	psimpl::simplify_douglas_peucker_n<2>(temp.begin(), temp.end(), vertices, back_inserter(temp2));
	vector <int> ret(temp2.size() / 2);
	for(int i = 0; i < ret.size(); i++) ret[i] = temp2[2 * i + 1] * WW + temp2[2 * i];
	return ret;
}
vector <int> simplify(const vector <int>& path, int WW, double tolerance){
	vector <double> temp(path.size()*2);
	for(int i = 0; i < path.size(); i++){
		temp[2 * i] = path[i] % WW;
		temp[2 * i + 1] = path[i] / WW;
	}
	vector <double> temp2;
	psimpl::simplify_douglas_peucker<2>(temp.begin(), temp.end(), tolerance, back_inserter(temp2));
	vector <int> ret(temp2.size() / 2);
	for(int i = 0; i < ret.size(); i++) ret[i] = temp2[2 * i + 1] * WW + temp2[2 * i];
	return ret;
}
double pathLength(const vector <int>& path, int WW){
	double ret = 0;
	int n = path.size();
	for(int i = 1; i < n; i++){
		int dx = path[i] % WW - path[i-1] % WW;
		int dy = path[i] / WW - path[i-1] / WW;
		ret += sqrt(dx * dx + dy * dy);
	}
	return ret;
}
double pathBackAngle(const vector <int>& path, int WW){
	double ret = 0;
	int n = path.size();
	if(n < 3) return 0;
	//if(n == 5) cerr << "---------" << endl;
	double prevDir = atan2(path[n-1] / WW - path[n-2] / WW, path[n-1] % WW - path[n-2] % WW) / PI * 180;
	for(int i = 1; i < n; i++){
		int dx = path[i] % WW - path[i-1] % WW;
		int dy = path[i] / WW - path[i-1] / WW;
		if(dx == 0 && dy == 0) continue;
		double dir = atan2(dy, dx) / PI * 180;
		double dif = dir - prevDir;
		while(dif > 180) dif -= 360;
		while(dif < -180) dif += 360;
		
		//if(n == 5) cerr << dx << " " << dy << " " << dir << " " << dif << endl;
		
		if(dif > 0) ret += dif;
		prevDir = dir;
	}
	//if(n == 5) cerr << "---------" << endl;
	return ret;	
}
double pathMaxMinAngle(const vector <int>& path, int WW){
	double ret = 0;
	int n = path.size();
	if(n < 3) return 0;
	double maxA = 0;
	double minA = 360;
	double prevDir = atan2(path[n-1] / WW - path[n-2] / WW, path[n-1] % WW - path[n-2] % WW) / PI * 180;
	//cerr << "----" << endl;
	for(int i = 1; i < n; i++){
		//cerr << "(" << path[i-1] % WW <<  "," << path[i-1] / WW << ")" << endl;
		int dx = path[i] % WW - path[i-1] % WW;
		int dy = path[i] / WW - path[i-1] / WW;
		if(dx == 0 && dy == 0) continue;
		double dir = atan2(dy, dx) / PI * 180;
		double dif = dir - prevDir;
		while(dif > 180) dif -= 360;
		while(dif < -180) dif += 360;
		double angle = 180 + dif;
		//cerr << dir << " " << dif << " " << angle << endl;
		maxA = max(maxA, angle);
		minA = min(minA, angle);
		prevDir = dir;
	}
	//cerr << maxA - minA;
	//cerr << "----" << endl;
	return maxA - minA;
}
double pathArea(const vector <int>& path, int WW){
	int n = path.size();
	if(n < 3) return 0;
	double a = path[0] % WW * (path[1] / WW - path[n - 2] / WW);
	for(int i = 1; i < n - 2; i++){
		a += path[i] % WW * (path[i + 1] / WW - path[i - 1] / WW);
	}
	a += path[n-2] % WW * (path[0] / WW - path[n - 3] / WW);
	return fabs(a / 2);
}
double pathBoundsArea(const vector <int>& path, int WW, double alpha){
	double min0 = 1e20;
	double max0 = -1e20;
	double min1 = 1e20;
	double max1 = -1e20;
	int n = path.size();
	for(int i = 0; i < n - 1; i++){
		double v0 = (path[i] % WW) * cos(alpha) + (path[i] / WW) * sin(alpha);
		double v1 = (path[i] % WW) * sin(alpha) - (path[i] / WW) * cos(alpha);
		min0 = min(min0, v0);
		max0 = max(max0, v0);
		min1 = min(min1, v1);
		max1 = max(max1, v1);
	}
	return (max0 - min0) * (max1 - min1);
}
vector <double> pathBounds(const vector <int>& path, int WW, double alpha){
	double min0 = 1e20;
	double max0 = -1e20;
	double min1 = 1e20;
	double max1 = -1e20;
	int n = path.size();
	for(int i = 0; i < n - 1; i++){
		double v0 = (path[i] % WW) * cos(alpha) + (path[i] / WW) * sin(alpha);
		double v1 = (path[i] % WW) * sin(alpha) - (path[i] / WW) * cos(alpha);
		min0 = min(min0, v0);
		max0 = max(max0, v0);
		min1 = min(min1, v1);
		max1 = max(max1, v1);
	}
	return {min0, max0, min1, max1};
}

struct Segment{
	int position;
	int area;
	int perimeter;
	int border;
	int rSum;
	//int gSum;
	//int bSum;
	//int bandSum[8];
	map <int, int> neigh;
	vector <int> refill;
	int xSum;
	int ySum;
	int lungPart[5];
	int prevSum;
	int nextSum;
	Segment(){
	}
	Segment(int pos, rgb color) : position(pos){
		area = 1;
		perimeter = 0;
		border = 0;
		rSum = color.r;
		//gSum = color.g;
		//bSum = color.b;
		//for(int i = 0; i < 8; i++) bandSum[i] = 0;
		xSum = 0;
		ySum = 0;
		prevSum = 0;
		nextSum = 0;
		for(int i = 0; i < 5; i++) lungPart[i] = 0;
	}
};

struct Slice{
	double x0;
	double y0;
	double z;
	double dx;
	double dy;
	double dz;
	Slice(){}
	Slice(double xx0, double yy0, double zz, double ddx, double ddy, double ddz) : x0(xx0), y0(yy0), z(zz), dx(ddx), dy(ddy), dz(ddz){} 
};

Slice extractAuxiliaryData(string folder, string scan, string sliceId){
	ifstream in((folder + scan + "/auxiliary/" + sliceId + ".dat").c_str());
	Slice slice;
	string line;
	while(!in.eof()){
		getline(in, line);
		if(!in.eof()){
			vector <string> row = splitBy(line, ',');
			if(!row.empty()){
				if(row[0] == "(0020.0032)"){
					slice.x0 = atof(row[1].c_str());
					slice.y0 = atof(row[2].c_str());
					slice.z = atof(row[3].c_str());
				}
				else if(row[0] == "(0028.0030)"){
					slice.dx = atof(row[1].c_str());
					slice.dy = atof(row[2].c_str());
				}
				else if(row[0] == "(0018.0050)"){
					slice.dz = atof(row[1].c_str());
				}
			}
		}
	}
	in.close();
	return slice;
}

pair<double, double> pixelToMm(double px, double py, Slice slice){
	double x = (px * slice.dx) + slice.x0;
	double y = (py * slice.dy) + slice.y0;
	return make_pair(x, y);
}
inline string pixelToMmStr(double px, double py, Slice slice){
	return SSTRF(px * slice.dx + slice.x0) + "," + SSTRF(py * slice.dy + slice.y0);
}
struct AreaInfo{
	int x0;
	int x1;
	int y0;
	int y1;
	vector <int> rx0;
	vector <int> rx1;
	AreaInfo(){
		x0 = W;
		x1 = -1;
		y0 = H;
		y1 = -1;
		rx0 = vector <int>(H, W);
		rx1 = vector <int>(H, -1);	
	}
	AreaInfo(int pixel){
		x0 = pixel % W;
		x1 = pixel % W;
		y0 = pixel / W;
		y1 = pixel / W;
		rx0 = vector <int>(H, W);
		rx1 = vector <int>(H, -1);
		rx0[y0] = x0;
		rx1[y0] = x0;
	}
	void update(int pixel){
		int x = pixel % W;
		int y = pixel / W;
		x0 = min(x0, x);
		x1 = max(x1, x);
		y0 = min(y0, y);
		y1 = max(y1, y);
		rx0[y] = min(rx0[y], x);
		rx1[y] = max(rx1[y], x);
	}
	void addEdge(int pixel1, int pixel2){
		if(pixel1 / W > pixel2 / W) swap(pixel1, pixel2);
		update(pixel1);
		update(pixel2);
		for(int y = pixel1 / W + 1; y < pixel2 / W; y++){
			int x = pixel1 % W + round(double(y - pixel1 / W) * (pixel2 % W - pixel1 % W) / (pixel2 / W - pixel1 / W));
			rx0[y] = min(rx0[y], x);
			rx1[y] = max(rx1[y], x);	
		}
	}

};

inline int cross(int O, int A, int B){
	return (A % W - O % W) * (B / W - O / W) - (A / W - O / W) * (B % W - O % W);
}

vector <int> convexHull(vector <int>& P){
	int n = P.size();
	if(n == 0) return P;
	int k = 0;
	vector <int> H(2 * n);

	sort(P.begin(), P.end(),  [&](int aa, int bb){return aa % W < bb % W || (aa % W == bb % W && aa / W < bb / W);});

	for(int i = 0; i < n; ++i){
		while(k >= 2 && cross(H[k-2], H[k-1], P[i]) <= 0) k--;
		H[k++] = P[i];
	}

	for (int i = n-2, t = k+1; i >= 0; i--){
		while (k >= t && cross(H[k-2], H[k-1], P[i]) <= 0) k--;
		H[k++] = P[i];
	}

	H.resize(k-1);
	return H;
}

int lungSize[5];
double lungPosition[4];
vector <char> detectLungs(image<rgb> *input){
	int WH = W * H;
	vector <int> area(WH, -1);
	int A = 0;
	vector <pair <int, int> > siz;
	vector <AreaInfo> areaInfo;
	
	for(int start = 0; start < WH; start++){
		if(area[start] == -1 && input->data[start].r < DARK){
			area[start] = A;
			siz.push_back(make_pair(A, 0));
			areaInfo.push_back(AreaInfo(start));
			vector <int> stack(1, start);
			while(!stack.empty()){
				int next = stack.back();
				stack.pop_back();
				siz[A].second++;
				areaInfo.back().update(next);
				if(next % W > 0 && area[next - 1] == -1 && input->data[next - 1].r < DARK){
					area[next - 1] = A;
					stack.push_back(next - 1);
				}
				if(next % W < W - 1 && area[next + 1] == -1 && input->data[next + 1].r < DARK){
					area[next + 1] = A;
					stack.push_back(next + 1);
				}
				if(next / W > 0 && area[next - W] == -1 && input->data[next - W].r < DARK){
					area[next - W] = A;
					stack.push_back(next - W);
				}
				if(next / W < H - 1 && area[next + W] == -1 && input->data[next + W].r < DARK){
					area[next + W] = A;
					stack.push_back(next + W);
				}
			}
			A++;
		}
	}
	sort(siz.begin(), siz.end(), [&](pair <int, int> aa, pair <int, int> bb){return aa.second > bb.second;});
	vector <int> rank(A);
	for(int i = 0; i < A; i++) rank[siz[i].first] = i;
	int lung[2];
	lung[0] = -2;
	lung[1] = -2;
	for(int i = 0; i < A; i++){
		if(lung[0] >= 0 && lung[1] >= 0) break;
		AreaInfo & ai = areaInfo[siz[i].first];
		if(lung[0] == -2 && ai.x0 < 3 * W / 7 && ai.y0 < 2 * H / 3 && ai.y1 < 6 * H / 7 && ai.y0 > H / 10 && (ai.y0 < H/2 || ai.x1 - ai.x0 < W / 2) ) lung[0] = siz[i].first;
		if(lung[1] == -2 && ai.x1 > 4 * W / 7 && ai.y0 < 2 * H / 3 && ai.y1 < 6 * H / 7 && ai.y0 > H / 10 && (ai.y0 < H/2 || ai.x1 - ai.x0 < W / 2) ) lung[1] = siz[i].first;
		if(lung[0] == lung[1] && lung[0] >= 0){
			if(ai.x0 + ai.x1 > W) lung[0] = -2;
			else lung[1] = -2;
		}
	}
	
	sort(siz.begin(), siz.end());
	int sizeL = 0;
	int sizeR = 0;
	int x0L = 0;
	int x1L = 0;
	int x0R = 0;
	int x1R = 0;
	if(lung[0] >= 0){
		sizeL = siz[lung[0]].second;
		x0L = areaInfo[lung[0]].x0;
		x1L = areaInfo[lung[0]].x1;
	}
	if(lung[1] >= 0){
		sizeR = siz[lung[1]].second;
		x0R = areaInfo[lung[1]].x0;
		x1R = areaInfo[lung[1]].x1;
	}
	if(3 * sizeL < sizeR && x1R - x0R > 0.3 * W){
		bool remove = false;
		if(sizeL > 0 && areaInfo[lung[0]].x1 < areaInfo[lung[1]].x0) remove = true;
		int newL = A;
		int newR = A + 1;
		areaInfo.push_back(AreaInfo());
		areaInfo.push_back(AreaInfo());
		for(int i = 0; i < WH; i++){
			if(area[i] == lung[0] && !remove){
				area[i] = newL;
				areaInfo[newL].update(i);
			} else if(area[i] == lung[1]){
				if(2 * (i % W) < x0R + x1R){
					area[i] = newL;
					areaInfo[newL].update(i);
				} else{
					area[i] = newR;
					areaInfo[newR].update(i);
				}
			}
		}
		lung[0] = newL;
		lung[1] = newR;
	} else if(3 * sizeR < sizeL && x1L - x0L > 0.3 * W){
		bool remove = false;
		if(sizeR > 0 && areaInfo[lung[1]].x0 > areaInfo[lung[0]].x1) remove = true;
		int newL = A;
		int newR = A + 1;
		areaInfo.push_back(AreaInfo());
		areaInfo.push_back(AreaInfo());
		for(int i = 0; i < WH; i++){
			if(area[i] == lung[1] && !remove){
				area[i] = newR;
				areaInfo[newR].update(i);
			} else if(area[i] == lung[0]){
				if(2 * (i % W) > x0L + x1L){
					area[i] = newR;
					areaInfo[newR].update(i);
				} else{
					area[i] = newL;
					areaInfo[newL].update(i);
				}
			}
		}
		lung[0] = newL;
		lung[1] = newR;	
	}
	if(EXPAND && lung[0] >= 0 && lung[1] >= 0){
		AreaInfo & aiL = areaInfo[lung[0]];
		AreaInfo & aiR = areaInfo[lung[1]];
		if(aiL.y1 + QEXPAND * (aiL.y1 - aiL.y0) < aiR.y1 && aiL.y1 > aiR.y0){
			for(int y = aiL.y1 + 1; y < aiR.y1; y++){
				int x0 = round(aiL.x0 +  double(aiR.x1 - aiR.rx1[y]) * (aiL.x1 - aiL.x0 + 1) / (aiR.x1 - aiR.x0 + 1));
				int x1 = round(aiL.x0 +  double(aiR.x1 - aiR.rx0[y]) * (aiL.x1 - aiL.x0 + 1) / (aiR.x1 - aiR.x0 + 1));
				if(area[W * y + x0] != lung[1]){
					area[W * y + x0] = lung[0];
					aiL.update(W * y + x0);
				}
				if(area[W * y + x1] != lung[1]){
					area[W * y + x1] = lung[0];
					aiL.update(W * y + x1);
				}
			}
		}
		else if(aiR.y1 + QEXPAND * (aiR.y1 - aiR.y0) < aiL.y1 && aiR.y1 > aiL.y0){
			for(int y = aiR.y1 + 1; y < aiL.y1; y++){
				int x0 = round(aiR.x0 +  double(aiL.x1 - aiL.rx1[y]) * (aiR.x1 - aiR.x0 + 1) / (aiL.x1 - aiL.x0 + 1));
				int x1 = round(aiR.x0 +  double(aiL.x1 - aiL.rx0[y]) * (aiR.x1 - aiR.x0 + 1) / (aiL.x1 - aiL.x0 + 1));
				if(area[W * y + x0] != lung[0]){
					area[W * y + x0] = lung[1];
					aiR.update(W * y + x0);
				}
				if(area[W * y + x1] != lung[0]){
					area[W * y + x1] = lung[1];
					aiR.update(W * y + x1);
				}
			}
		}
		if(aiL.y0 - QEXPAND * (aiL.y1 - aiL.y0) > aiR.y0 && aiL.y0 < aiR.y1){
			for(int y = aiR.y0; y < aiL.y0 - 1; y++){
				int x0 = round(aiL.x0 +  double(aiR.x1 - aiR.rx1[y]) * (aiL.x1 - aiL.x0 + 1) / (aiR.x1 - aiR.x0 + 1));
				int x1 = round(aiL.x0 +  double(aiR.x1 - aiR.rx0[y]) * (aiL.x1 - aiL.x0 + 1) / (aiR.x1 - aiR.x0 + 1));
				if(area[W * y + x0] != lung[1]){
					area[W * y + x0] = lung[0];
					aiL.update(W * y + x0);
				}
				if(area[W * y + x1] != lung[1]){
					area[W * y + x1] = lung[0];
					aiL.update(W * y + x1);
				}
			}
		}
		else if(aiR.y0 - QEXPAND * (aiR.y1 - aiR.y0) > aiL.y0 && aiR.y0 < aiL.y1){
			for(int y = aiL.y0; y < aiR.y0 - 1; y++){
				int x0 = round(aiR.x0 +  double(aiL.x1 - aiL.rx1[y]) * (aiR.x1 - aiR.x0 + 1) / (aiL.x1 - aiL.x0 + 1));
				int x1 = round(aiR.x0 +  double(aiL.x1 - aiL.rx0[y]) * (aiR.x1 - aiR.x0 + 1) / (aiL.x1 - aiL.x0 + 1));
				if(area[W * y + x0] != lung[0]){
					area[W * y + x0] = lung[1];
					aiR.update(W * y + x0);
				}
				if(area[W * y + x1] != lung[0]){
					area[W * y + x1] = lung[1];
					aiR.update(W * y + x1);
				}
			}
		}
	}
	
	
	vector <int> hull[3];
	for(int side = 0; side < 2; side++) if(lung[side] >= 0){
		AreaInfo & ai = areaInfo[lung[side]];
		for(int y = 0; y < H; y++){
			if(ai.rx0[y] <= ai.rx1[y]){
				hull[side].push_back(y * W + ai.rx0[y]);
				hull[side].push_back(y * W + ai.rx1[y]);
				hull[2].push_back(y * W + ai.rx0[y]);
				hull[2].push_back(y * W + ai.rx1[y]);
			}
		}
		hull[side] = convexHull(hull[side]);
	}
	hull[2] = convexHull(hull[2]);
	
	AreaInfo hullInfo[3];
	for(int side = 0; side < 3; side++){
		for(int i = 0; i < hull[side].size(); i++){
			hullInfo[side].addEdge(hull[side][i], hull[side][(i + 1) % hull[side].size()]);
		}
	}
	
	vector <char> isHull(WH, 0);
	
	for(int side = 2; side >= 0; side--){
		lungSize[side + 2] = 0;
		for(int y = 0; y < H; y++){
			for(int x = hullInfo[side].rx0[y]; x <= hullInfo[side].rx1[y]; x++){
				isHull[W * y + x] = side + 3;
				lungSize[side + 2]++;
			}
		}
	}

	lungSize[0] = 0;
	lungSize[1] = 0;
	lungPosition[0] = 0;
	lungPosition[1] = 0;
	lungPosition[2] = 0;
	lungPosition[3] = 0;
	for(int i = 0; i < WH; i++){
		if(!isHull[i] || isHull[i] == 5) input->data[i] = INTRGB(0x000000);
		if(area[i] == lung[0]){
			isHull[i] = 1;
			lungSize[0]++;
			lungPosition[0] += i % W;
			lungPosition[1] += i / W;
		}
		else if(area[i] == lung[1]){
			isHull[i] = 2;
			lungSize[1]++;
			lungPosition[2] += i % W;
			lungPosition[3] += i / W;
		}
	}
	if(lungSize[0] > 0){
		lungPosition[0] /= lungSize[0];
		lungPosition[1] /= lungSize[0];
	} else{
		lungPosition[0] = -1000;
		lungPosition[1] = -1000;
	}
	if(lungSize[1] > 0){
		lungPosition[2] /= lungSize[1];
		lungPosition[3] /= lungSize[1];
	} else{
		lungPosition[2] = -1000;
		lungPosition[3] = -1000;
	}
	
	return isHull;
}

int main(int argc, char* argv[]){
	
	string phase = argc > 1 ? argv[1] : "train";
	int step = argc > 2 ? atoi(argv[2]) : 1;
	float sigma = argc > 3 ? atof(argv[3]) : 0.5;
	float k = argc > 4 ? atof(argv[4]) : 50;//100;
	int min_size = argc > 5 ? atoi(argv[5]) : 70; //50;
	int vertices = argc > 6 ? atoi(argv[6]) : 30;
	string run = argc > 7 ? argv[7] : "";

	int pre = 7;

	string trainFolder = argc > 1 + pre ? argv[1 + pre] : "example_extracted_2";
	string testFolder = argc > 2 + pre ? argv[2 + pre] : "provisional_extracted_2";
	string outputFile = argc > 3 + pre ? argv[3 + pre] : "result"+SSTR(step)+"_"+SSTR(sigma)+"_"+SSTR(k)+"_"+SSTR(min_size)+"_"+SSTR(vertices)+".csv";
	
	string trainFolderDos = trainFolder;
	string testFolderDos = testFolder;
	
	if(trainFolderDos.back() != '\\') trainFolderDos.push_back('\\');
	if(testFolderDos.back() != '\\') testFolderDos.push_back('\\');
		
	if(trainFolder.back() != '/') trainFolder.push_back('/');
	if(testFolder.back() != '/') testFolder.push_back('/');

	vector <string> trainScans = listdirdir(trainFolder);
	vector <string> testScans = listdirdir(testFolder);
	
	vector <string> trainFiles;
	vector <string> testFiles;
	vector <string> trainScanId;
	vector <string> testScanId;
	vector <string> trainSliceId;
	vector <string> testSliceId;
	vector <int> trainScanSize;
	vector <int> testScanSize;
	
	for(int i = 0; i < trainScans.size(); i++){
		//vector <string> slices = listdir(trainFolder + trainScans[i] + "/pngs");
		//for(int j = 0; j < slices.size(); j++) trainFiles.push_back(trainScans[i] + "/pngs/" + slices[j]);
		vector <string> slices = listdir(trainFolder + trainScans[i] + "/ppm");
		for(int j = 0; j < slices.size(); j++){
			trainFiles.push_back(trainScans[i] + "/ppm/" + slices[j]);
			trainScanId.push_back(trainScans[i]);
			trainSliceId.push_back(slices[j].replace(slices[j].end() - 4, slices[j].end(), ""));
			trainScanSize.push_back(slices.size());
		}
		/*for(int j = 0; j < slices.size(); j++){
				clog << i << " " << j << "\r";
			system(("magick convert " + trainFolder + trainScans[i] + "/pngs/" + SSTR(j + 1) + ".png " + " -define png:color-type=2 -depth 8 " + trainFolder + trainScans[i] + "/ppm/" + SSTR(j + 1) + ".ppm ").c_str());
		}*/
		//system(("mkdir " + trainFolderDos + trainScans[i] + "\\ppm").c_str());
	}
	for(int i = 0; i < testScans.size(); i++){
		//vector <string> slices = listdir(testFolder + testScans[i] + "/pngs");
		//for(int j = 0; j < slices.size(); j++) testFiles.push_back(testScans[i] + "/pngs/" + slices[j]);
		vector <string> slices = listdir(testFolder + testScans[i] + "/ppm");
		for(int j = 0; j < slices.size(); j++){
			testFiles.push_back(testScans[i] + "/ppm/" + slices[j]);
			testScanId.push_back(testScans[i]);	
			testSliceId.push_back(slices[j].replace(slices[j].end() - 4, slices[j].end(), ""));
			testScanSize.push_back(slices.size());
		}
		/*for(int j = 0; j < slices.size(); j++){
				clog << i << " " << j << "\r";
			system(("magick convert " + testFolder + testScans[i] + "/pngs/" + SSTR(j + 1) + ".png " + " -define png:color-type=2 -depth 8 " + testFolder + testScans[i] + "/ppm/" + SSTR(j + 1) + ".ppm ").c_str());
		}*/
		//system(("mkdir " + testFolderDos + testScans[i] + "\\ppm").c_str());
	}
	
	cout << "Train files: " << trainFiles.size() << endl;
	cout << " Test files: " << testFiles.size() << endl;
	
	string command;
	#if LINUX==0
	string redir2 = " 2>nul";
	#else
	string redir2 = " 2>/dev/null";
	#endif
	#if LINUX==0
	string redir = " >nul";
	#else
	string redir = " >/dev/null";
	#endif
	
	
	double start = getTime();
	
	vector <Tree> randomForest(TREES);
	
	//unordered_map <string, int> trainId;
	//unordered_map <string, double> trainFr;
	
	if(phase == "test" || run == "1"){
		ifstream in("traindata.csv");
		string line;
		int ln = 0;
		string prevId = "";
		int num = 0;
		while(!in.eof()){
			if(ln % 100000 == 0) clog << ln << "\r";
			getline(in,line);
			if(!in.eof()){
				vector <string> row = splitBy(line,',');
				vector <float> feat(FEATURES + 1, 0);
				if(row.size() < FEATURES + 3) cerr << "Incorrect number of items in traindata.csv" << endl;
				else{
					for(int i = 0; i < FEATURES + 1; i++) feat[i] = atof(row[i + 2].c_str());
				}
				if(!USEMSE) feat.back() = feat.back() > ATLEAST ? 1 : 0;
				trainData.push_back(Item(ln, feat));
				if(prevId != row[0]+ "," + row[1]){
					prevId = row[0]+ "," + row[1];
					num = 0;
				}
				else num++;
				//trainId[prevId + "," + SSTR(num)] = ln;
				ln++;
			}
		}
		clog << trainData.size() << " segments found in traindata." << endl;
		in.close();
		
		featureScore.clear();
		featureScoreC.clear();
		featureScore.resize(FEATURES,0);
		featureScoreC.resize(FEATURES,0);
		ofstream pred("prediktors.txt");
		double tic = getTime();
		for(int j = 0; j < TREES; j++){
			randomForest[j] = USEMSE ? buildTreeMSE() : buildTree();
			if(j % 10 == 0) clog << j + 1 << " trees done...\r";
		}
		clog << endl;
		clog << "Average per tree: " << (getTime() - tic) / TREES << " sec." << endl;
		vector <pair <int,int> > stat;
		for(int i = 0; i < FEATURES; i++) stat.push_back(make_pair(featureScore[i], i));
		sort(stat.begin(), stat.end());
		int len = log10(stat.back().first + 0.1) + 2;
		vector <pair <int,int> > statC;
		for(int i = 0; i < FEATURES; i++) statC.push_back(make_pair(featureScoreC[i], i));
		sort(statC.begin(),statC.end());
		int lenC = log10(statC.back().first + 0.1) + 2;
		for(int i = FEATURES - 1; i >= 0; i--){
			pred << int2len(stat[i].first, len) << " " << string2len(SSTR(stat[i].second), 3) << "   |   " << int2len(statC[i].first, lenC) << " " << statC[i].second << endl;
		}
		pred.close();
		
		#if OOB==1
		tic = getTime();
		ifstream in2("result.csv");
		ofstream ou2("result_oob.csv");
		for(int i = 0; i < trainData.size(); i++){
			if(i % 10000 == 0) clog << i << "\r";
			getline(in2, line);
			if(trainData[i].oobCount > 0) trainData[i].oobResult /= trainData[i].oobCount;
			else cerr << "trainData[" << i << "].oobCount == 0 !!!" << endl;
			ou2 << trainData[i].oobResult << "," << line << endl;
		}
		in2.close();
		ou2.close();
		/*for(auto it = trainId.begin(); it != trainId.end(); it++){
			trainFr[it->first] = trainData[it->second].oobResult;
		}*/
		sort(trainData.begin(), trainData.end(), [&](Item& aa, Item& bb){return aa.oobResult > bb.oobResult;});
		clog << "Sorting OOB results took " << getTime() - tic << " seconds." << endl;
		tic = getTime();
		int tp = 0;
		double allP = 245763;
		double maxScore = 0;
		double threshold = 0;
		double bestI = 0;
		ofstream sc("scorelog.csv");
		for(int i = 0; i < trainData.size(); i++){
			if(i % 100000 == 0) clog << i << "\r";
			if(trainData[i].result > ATLEAST) tp++;
			double precision = tp / (i + 1.0);
			double recall = tp / allP;
			double fScore = tp > 0 ? 2 * precision * recall / (precision + recall) : 0;
			if(fScore > maxScore){
				maxScore = fScore;
				bestI = i + 1;
				threshold = trainData[i].oobResult;
			}
			sc << i << "," << trainData[i].result << "," << trainData[i].oobResult << "," << precision << "," << recall << "," << fScore << "\n";
		}
		sc.close();
		clog << "Writing OOB results took " << getTime() - tic << " seconds." << endl;
		clog << "Best threshold: " << threshold << endl;
		clog << "         Ratio: " << bestI / trainData.size() << endl;
		clog << "  Best F_score: " << maxScore << endl;
		#endif
	}
	
	//ofstream out(outputFile.c_str());
	int outFiles = (phase == "train" && run != "1") ? 1 : THRESHOLD.size();
	vector <ofstream> out(outFiles);
	for(int i = 0; i < outFiles; i++) out[i].open((outputFile + SSTR(i)).c_str());
	
	ofstream outf;
	if(phase == "train") outf.open("features.csv");
	else outf.open("features_test.csv");
	
	//if(breaking == 0) breaking = trainFiles.size();

	int totalSegments = 0;
	//int returnedSegments = 0;
	vector <int> returnedSegments(outFiles, 0);
	int joinings = 0;
	for(int i = 0; i < ( phase == "train" ? trainFiles.size() : testFiles.size() ); i++){
		if(i % step == 0) clog << i << "\r";
		string folder = phase == "train" ? trainFolder : testFolder;
		string scanId = phase == "train" ? trainScanId[i] : testScanId[i];
		string sliceId = phase == "train" ? trainSliceId[i] : testSliceId[i];
		string imageId = phase == "train" ? trainFiles[i] : testFiles[i];
		int scanSize = phase == "train" ? trainScanSize[i] : testScanSize[i];
		int sliceOrder = atoi(sliceId.c_str());
		imageId.replace(imageId.end() - 4, imageId.end(), "");
		imageId.replace(imageId.begin(), imageId.begin() + 6, "");
		if(i % step == 0){
			Slice slice = extractAuxiliaryData(folder, scanId, sliceId);

		//if(i < breaking){
			//if(phase == "train") command = "magick convert " + trainFolder + trainFiles[i] + " -define png:color-type=2 -depth 8 " +  "temp" + run + ".ppm" + redir2;
			//else command = "magick convert " + testFolder + testFiles[i] + " -define png:color-type=2 -depth 8 " +  "temp" + run + ".ppm" + redir2;
			//system(command.c_str());
			//image<rgb> *input = loadPPM(("temp" + run + ".ppm").c_str());
			string prevSlice = folder + scanId + "/ppm/" + (sliceOrder == 1 ? sliceId : SSTR(sliceOrder - 1)) + ".ppm";
			string nextSlice = folder + scanId + "/ppm/" + (sliceOrder == scanSize ? sliceId : SSTR(sliceOrder + 1)) + ".ppm";
			image<rgb> *input0 = loadPPM(prevSlice.c_str());
			image<rgb> *input = loadPPM((phase == "train" ? trainFolder + trainFiles[i] : testFolder + testFiles[i]).c_str());
			image<rgb> *input1 = loadPPM(nextSlice.c_str());
			
			vector <char> isHull = detectLungs(input);
			
			int numC; 
			image<rgb> *seg = segment_image(input, sigma, k, min_size, &numC);
			
			unordered_map <int, Segment> segments;
			//int W = seg->width();
			//int H = seg->height();
			int WW = W + 1;
			int HH = H + 1;
			int WH = W * H;
			nextColor = 0;
			
			for(int j = 0; j < WH; j++){
				if(segments.count(RGBINT(seg->data[j])) == 0){
					Segment& segment = segments[RGBINT(seg->data[j])];
					segment = Segment(j, input->data[j]);
				}
				else{
					Segment& segment = segments[RGBINT(seg->data[j])];
					segment.area++;
					segment.rSum += input->data[j].r;
					//segment.gSum += input->data[j].g;
					//segment.bSum += input->data[j].b;
				}
				Segment& segment = segments[RGBINT(seg->data[j])];
				if(j % W != 0 && RGBINT(seg->data[j - 1]) != RGBINT(seg->data[j])){
					segment.neigh[RGBINT(seg->data[j - 1])]++;
					segment.border++;
				}
				if(j % W != W - 1 && RGBINT(seg->data[j + 1]) != RGBINT(seg->data[j])){
					segment.neigh[RGBINT(seg->data[j + 1])]++;
					segment.border++;
				}
				if(j / W != 0 && RGBINT(seg->data[j - W]) != RGBINT(seg->data[j])){
					segment.neigh[RGBINT(seg->data[j - W])]++;
					segment.border++;
				}
				if(j / W != H - 1 && RGBINT(seg->data[j + W]) != RGBINT(seg->data[j])){
					segment.neigh[RGBINT(seg->data[j + W])]++;
					segment.border++;
				}
				if(j % W == 0) segment.border++;
				if(j % W == W - 1) segment.border++;
				if(j / W == 0) segment.border++;
				if(j / W == H - 1) segment.border++;
				segment.xSum += j % W;
				segment.ySum += j / W;
				segment.prevSum += input0->data[j].r;
				segment.nextSum += input1->data[j].r;
				if(isHull[j]) segment.lungPart[isHull[j] - 1]++;
			}
			
			//savePPM(seg, "temp_seg.ppm");
			//continue;
			
			
			int buildingId = 0;
			for(auto it = segments.begin(); it != segments.end(); ++it){
				buildingId++;
				rgb color = INTRGB(it->first);
				int start = it->second.position;
				vector <int> path = {(start / W) * WW + start % W, (start / W) * WW + start % W + WW};
				start = (start / W) * WW + start % W;
				int n = 1;
				//clog << " " << start/WW << " " << start%WW;
				while(path.back() != start && n < WH * 4){
					int test[3];
					if(path[n] - path[n - 1] == WW){
						test[0] = path[n] - 1;
						test[1] = path[n] + WW;
						test[2] = path[n] + 1;
					} else if(path[n] - path[n - 1] == -WW){
						test[0] = path[n] + 1;
						test[1] = path[n] - WW;
						test[2] = path[n] - 1;
					} else if(path[n] - path[n - 1] == 1){
						test[0] = path[n] + WW;
						test[1] = path[n] + 1;
						test[2] = path[n] - WW;
					} else{
						test[0] = path[n] - WW;
						test[1] = path[n] - 1;
						test[2] = path[n] + WW;
					}
					for(int ii = 0; ii < 3; ii++){
						if(test[ii] == path[n] + WW){
							if(path[n] % WW < W && path[n] / WW < H && imRef(seg, path[n] % WW, path[n] / WW) == color && (path[n] % WW == 0 || !(imRef(seg, path[n] % WW - 1, path[n] / WW) == color) ) ){
								path.push_back(test[ii]);
								//clog << "V";
								break;
							}
						} else if(test[ii] == path[n] - WW){
							if(path[n] % WW > 0 && path[n] / WW > 0 && imRef(seg, path[n] % WW - 1, path[n] / WW - 1) == color && (path[n] % WW == W || !(imRef(seg, path[n] % WW, path[n] / WW - 1) == color) ) ){
								path.push_back(test[ii]);
								//clog << "^";
								break;
							}
						} else if(test[ii] == path[n] + 1){
							if(path[n] % WW < W && path[n] / WW > 0 && imRef(seg, path[n] % WW, path[n] / WW - 1) == color && (path[n] / WW == H || !(imRef(seg, path[n] % WW, path[n] / WW) == color) ) ){
								path.push_back(test[ii]);
								//clog << ">";
								break;
							}
						}
						else{
							if(path[n] % WW > 0 && path[n] / WW < H && imRef(seg, path[n] % WW - 1, path[n] / WW) == color && (path[n] / WW == 0 || !(imRef(seg, path[n] % WW - 1, path[n] / WW - 1) == color) ) ){
								path.push_back(test[ii]);
								//clog << "<";
								break;
							}
						}
					}
					n++;
					if(path.size() != n + 1) cerr << "Path not extended!!! Unexpected situation..." << endl;
				}
				it->second.perimeter = n;
				//clog << " " << n << endl;
				vector <int> sPath;
				sPath.push_back(path[0]);
				for(int j = 1; j < path.size() - 1; j++){
					if((path[j] % WW != path[j - 1] % WW || path[j] % WW != path[j + 1] % WW) && (path[j] / WW != path[j - 1] / WW || path[j] / WW != path[j + 1] / WW)) sPath.push_back(path[j]);
				}
				sPath.push_back(path.back());
				
				vector <int> sPathT[4];
				double len[4];
				sPathT[0] = simplify(sPath, WW, 0.9);
				sPathT[1] = simplify(sPath, WW, 1.4);
				sPathT[2] = simplify(sPath, WW, 2.5);
				sPathT[3] = simplify(sPath, WW, 5);
				for(int j = 0; j < 4; j++) len[j] = pathLength(sPathT[j], WW);
				vector <int> sPathV[3];
				sPathV[0] = simplifyV(sPath, WW, 20);
				sPathV[1] = simplifyV(sPath, WW, 10);
				sPathV[2] = simplifyV(sPath, WW, 5);
				
				sPath = simplifyV(sPath, WW, vertices);
				double rMean = double(it->second.rSum)/it->second.area;
				//double gMean = double(it->second.gSum)/it->second.area;
				//double bMean = double(it->second.bSum)/it->second.area;
				
				vector <float> feat;
				feat.push_back(it->second.area);                   // 0
				feat.push_back(it->second.perimeter);              // 1
				feat.push_back(rMean);                             // 2
				//feat.push_back(gMean);
				//feat.push_back(bMean);
				feat.push_back(segments.size());                               // 3
				feat.push_back(it->second.perimeter/sqrt(it->second.area));    // 4
				for(int j = 0; j < 3; j++){
					feat.push_back(len[j]);                                    // 5, 7, 9
					feat.push_back(len[j] / sqrt(it->second.area));            // 6, 8, 10
				}
				feat.push_back(sPathT[2].size());                              // 11
				feat.push_back(sPathT[3].size());                              // 12
				for(int j = 0; j < 3; j++) feat.push_back(len[0] > 0 ? pathLength(sPathV[j], WW) / len[0] : 0);           // 13, 14, 15
				feat.push_back(pathBackAngle(sPath, WW));                                                                 // 16
				for(int j = 0; j < 3; j++) feat.push_back(pathBackAngle(sPathV[j], WW));                                  // 17, 18, 19
				feat.push_back(pathMaxMinAngle(sPathV[2], WW));                // 20
				
				double minDifr = 256;
				//double minDifg = 256;
				//double minDifb = 256;
				//double minDif = sqrt(3)*256;
				//double minDifB = sqrt(8)*256;
				for(auto it2 = it->second.neigh.begin(); it2 != it->second.neigh.end(); ++it2){
					Segment& neighbor = segments[it2->first];
					double rM = double(neighbor.rSum) / neighbor.area;
					//double gM = double(neighbor.gSum) / neighbor.area;
					//double bM = double(neighbor.bSum) / neighbor.area;
					//double dif = sqrt((rMean - rM) * (rMean - rM) + (gMean - gM) * (gMean - gM) + (bMean - bM) * (bMean - bM));
					minDifr = min(minDifr, fabs(rMean - rM));
					//minDifg = min(minDifg, fabs(gMean - gM));
					//minDifb = min(minDifb, fabs(bMean - bM));
					//minDif = min(minDif, dif);
				}
				feat.push_back(minDifr);                       // 21
				//feat.push_back(minDifg);
				//feat.push_back(minDifb);
				//feat.push_back(minDif);
				
				double pArea = pathArea(sPath, WW);

				double bestAngle = 0;
				double minArea = 1e30;
				for(double angle = 0; angle < PI/2; angle += PI/180){
					double area = pathBoundsArea(sPath, WW, angle);
					if(area < minArea){
						minArea = area;
						bestAngle = angle;
					}
				} 
				for(double angle = bestAngle - 2*PI/180; angle < bestAngle + 2*PI/180; angle += PI/1800){
					double area = pathBoundsArea(sPath, WW, angle);
					if(area < minArea){
						minArea = area;
						bestAngle = angle;
					}
				}
				//cerr << bestAngle*180/PI << " " << minArea << " " << pArea << endl;
				/*vector <double> bounds = pathBounds(sPath, WW, bestAngle);
				vector <double> retX(5);
				vector <double> retY(5);
				retX[0] = bounds[0] * cos(bestAngle) + bounds[2] * sin(bestAngle);
				retY[0] = -bounds[2] * cos(bestAngle) + bounds[0] * sin(bestAngle);
				retX[1] = bounds[0] * cos(bestAngle) + bounds[3] * sin(bestAngle);
				retY[1] = -bounds[3] * cos(bestAngle) + bounds[0] * sin(bestAngle);
				retX[2] = bounds[1] * cos(bestAngle) + bounds[3] * sin(bestAngle);
				retY[2] = -bounds[3] * cos(bestAngle) + bounds[1] * sin(bestAngle);
				retX[3] = bounds[1] * cos(bestAngle) + bounds[2] * sin(bestAngle);
				retY[3] = -bounds[2] * cos(bestAngle) + bounds[1] * sin(bestAngle);
				retX[4] = retX[0];
				retY[4] = retY[0];
				*/
				
				feat.push_back(pArea / minArea);            // 22
				
				//double Q = sqrt(1 + (pArea < 100 ? 0.463742023702703 : (pArea > 1000 ? -0.328177012792306 : 1.090303293651970e-06 * pArea * pArea - 0.002079243663567 * pArea + 0.660763357122901)));
				double Gx = double(it->second.xSum) / it->second.area;
				double Gy = double(it->second.ySum) / it->second.area;
				/*vector <double> retX(sPath.size());
				vector <double> retY(sPath.size());
				for(int j = 0; j < sPath.size(); j++){
					retX[j] = Gx + 0.5 + Q * (sPath[j] % WW - Gx - 0.5);
					retY[j] = Gy + 0.5 + Q * (sPath[j] / WW - Gy - 0.5) + 0.5;
					retX[j] = round(10 * retX[j]) / 10;
					retY[j] = round(10 * retY[j]) / 10;
					if(retX[j] < 0) retX[j] = 0;
					if(retX[j] > W) retX[j] = W; 
					if(retY[j] < 0) retY[j] = 0;
					if(retY[j] > H) retY[j] = H; 
				}*/
				/*double Q = sqrt(pArea / minArea);
				for(int j = 0; j < retX.size(); j++){
					retX[j] = Gx + Q * (retX[j] - Gx);
					retY[j] = Gy + Q * (retY[j]- Gy);
					retX[j] = round(10 * retX[j]) / 10;
					retY[j] = round(10 * retY[j]) / 10;
					if(retX[j] < 0) retX[j] = 0;
					if(retX[j] > W) retX[j] = W; 
					if(retY[j] < 0) retY[j] = 0;
					if(retY[j] > H) retY[j] = H; 
				}*/
				
				feat.push_back(sliceOrder);                            // 23
				feat.push_back(double(sliceOrder) / scanSize);         // 24
				feat.push_back(slice.z);                               // 25
				feat.push_back(slice.dx);                              // 26
				feat.push_back(slice.x0);                              // 27
				feat.push_back(slice.y0);                              // 28
				feat.push_back(Gx);                                    // 29
				feat.push_back(Gy);                                    // 30
				pair <double, double> G = pixelToMm(Gx, Gy, slice);
				feat.push_back(G.first);                               // 31
				feat.push_back(G.second);                              // 32
				feat.push_back(it->second.area * slice.dx * slice.dy); // 33 
				feat.push_back(it->second.perimeter * (slice.dx + slice.dy) / 2);    // 34
				feat.push_back(it->second.neigh.size());                             // 35
				
				for(int j = 0; j < 5; j++) feat.push_back(lungSize[j]);                                  // 36, 37, 38, 39, 40
				feat.push_back(fabs(lungSize[0] - lungSize[1]) / max(1, lungSize[0] + lungSize[1]));         // 41
				feat.push_back(double(lungSize[0] + lungSize[1]) / max(1, lungSize[2] + lungSize[3]));       // 42
				feat.push_back(double(it->second.lungPart[0] + it->second.lungPart[1]) / it->second.area);   // 43
				feat.push_back(double(it->second.lungPart[0] + it->second.lungPart[1] + it->second.lungPart[2] + it->second.lungPart[3]) / it->second.area);   // 44
				feat.push_back(double(it->second.lungPart[4]) / it->second.area);                            // 45
				
				double dL = (Gx - lungPosition[0]) * (Gx - lungPosition[0]) + (Gy - lungPosition[1]) * (Gy - lungPosition[1]);
				double dR = (Gx - lungPosition[2]) * (Gx - lungPosition[2]) + (Gy - lungPosition[3]) * (Gy - lungPosition[3]);
				if(dL < dR){
					feat.push_back(Gx - lungPosition[0]);                     // 46
					feat.push_back(Gy - lungPosition[1]);                     // 47
				} else{
					feat.push_back(lungPosition[2] - Gx);
					feat.push_back(Gy - lungPosition[3]);
				}
				feat.push_back(lungPosition[2] - lungPosition[0]);     // 48
				feat.push_back(lungPosition[3] - lungPosition[1]);     // 49
				
				double prevMean = double(it->second.prevSum)/it->second.area;
				double nextMean = double(it->second.nextSum)/it->second.area;
				feat.push_back(prevMean);                               // 50
				feat.push_back(nextMean);                               // 51
				feat.push_back(prevMean + nextMean);                    // 52
				feat.push_back(rMean - prevMean);                    // 53
				feat.push_back(nextMean - rMean);                    // 54
				feat.push_back(prevMean + nextMean - 2 * rMean);     // 55
			
				
				double fr = 0;
				if(phase == "test") fr = forestAssignResult(randomForest, Item(0, feat));
				//else if(run == "1") fr = trainFr[scanId + "," + sliceId + "," + SSTR(buildingId - 1)];
				outf << scanId << "," << sliceId;
				for(int j = 0; j < feat.size(); j++) outf << "," << feat[j];
				if(phase == "test" || run == "1") outf << "," << fr;
				outf << endl;
				for(int of = 0; of < outFiles; of++){
					if((phase == "train" && run != "1") || fr >= THRESHOLD[of]){
						returnedSegments[of]++;
						out[of] << scanId << "," << sliceId;
						for(int j = 0; j < sPath.size(); j++){
							out[of] << ",";
							//out << sPath[j] % WW << "," << sPath[j] / WW;
							out[of] << pixelToMmStr(sPath[j] % WW, sPath[j] / WW, slice);
						}
						out[of] << endl;
					}
				}
				totalSegments++;
			}
			
			delete input;
			delete input0;
			delete input1;
			delete seg;

		}
		//else out << imageId << ",-1,POLYGON EMPTY,1" << endl;
	}
	for(int of = 0; of < outFiles; of++){
		clog << "  " << of << " " << THRESHOLD[of] << ": " << returnedSegments[of] << " / " << totalSegments << " = " << double(returnedSegments[of])/totalSegments << endl;
	}
	clog << endl;
	clog << getTime() - start << endl;
	
	//out.close();
	for(int i = 0; i < outFiles; i++) out[i].close();
	outf.close();
	
	//system(("convert temp.ppm temp.png " + redir2).c_str());
	//system(("convert temp_seg.ppm temp_seg.png " + redir2).c_str());
	
	return 0;
}

