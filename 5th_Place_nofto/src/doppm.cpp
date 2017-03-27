#include "location.h" 
#include <bits/stdc++.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <dirent.h>
#define SSTR( x ) dynamic_cast< std::ostringstream & >( ( std::ostringstream() << std::dec << x ) ).str()
#define SSTRF( x ) dynamic_cast< std::ostringstream & >( ( std::ostringstream() << fixed << setprecision(2) << x ) ).str()
#define RGBINT( x ) ( (x.r << 16) + (x.g << 8) + x.b )

using namespace std;


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

int main(int argc, char* argv[]){
	
	string folder = argc > 1 ? argv[1] : "example_extracted_sample";
	
	string folderDos = folder;
	if(folderDos.back() != '\\') folderDos.push_back('\\');
	if(folder.back() != '/') folder.push_back('/');
	vector <string> scans = listdirdir(folder);
	
	vector <string> trainFiles;
	vector <string> testFiles;
	vector <string> trainScanId;
	vector <string> testScanId;
	vector <string> trainSliceId;
	vector <string> testSliceId;
	vector <int> trainScanSize;
	vector <int> testScanSize;
	
	for(int i = 0; i < scans.size(); i++){
		system(("mkdir " + folder + scans[i] + "/ppm").c_str());              // ADDED
		vector <string> slices = listdir(folder + scans[i] + "/pngs");
		for(int j = 0; j < slices.size(); j++){
			clog << i << " " << j << "\r";
			system((
				"magick convert " + folder + scans[i] + "/pngs/" + SSTR(j + 1) + ".png " + "-modulate 2000% -define png:color-type=2 -depth 8 "
				+ folder + scans[i] + "/ppm/" + SSTR(j + 1) + ".ppm ").c_str());
		}
		clog << "       \r";
		//system(("mkdir " + trainFolderDos + trainScans[i] + "\\ppm").c_str());
	}
	
	return 0;
}

