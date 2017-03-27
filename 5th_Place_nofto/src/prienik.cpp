#include <bits/stdc++.h>
#include <sys/time.h>
#include <sys/stat.h>
#define SSTR( x ) dynamic_cast< std::ostringstream & >( ( std::ostringstream() << std::dec << x ) ).str()

#include "pia.c"

using namespace std;

double getTime() {
	unsigned long long timelo, timehi;
    __asm__ volatile ("rdtsc" : "=a" (timelo), "=d" (timehi));
    return ((timehi << 32) + timelo) / 2.5e9;
}

string trim(const string& str,const string& whitespace=" \t\r\n"){
    size_t strBegin = str.find_first_not_of(whitespace);
    if (strBegin == string::npos) return "";
    size_t strEnd = str.find_last_not_of(whitespace);
    size_t strRange = strEnd - strBegin + 1;
    return str.substr(strBegin, strRange);
}
vector <string> splitBy(const string& text,char by){   // split string by by
	vector <string> vys;
	stringstream ss(text);
    string word;
    while(getline(ss,word,by)){
        vys.push_back(word);
    }
    return vys;
}
vector <string> splitBy2(const string& text,char by){   // split string by by
	vector <string> vys;
	stringstream ss(text);
    string word;
    int c = 0;
    while(getline(ss,word,by)){
        c++;
		vys.push_back(word);
        if(c == 2) by = '#';
    }
    return vys;
}
inline bool fileExists(const string& name){
	struct stat buffer;   
	return (stat(name.c_str(), &buffer) == 0); 
}
double area(const vector <point_t>& polygon){
	if(polygon.size() < 3) return 0;
	double a = polygon[0].x * (polygon[1].y - polygon[polygon.size() - 1].y);
	for(int i = 1; i < polygon.size() - 1; i++){
		a += polygon[i].x * (polygon[i+1].y - polygon[i-1].y);
	}
	a += polygon[polygon.size() - 1].x * (polygon[0].y - polygon[polygon.size() - 2].y);
	return fabs(a / 2);
}


int main(int argc, char* argv[]){
	
	double start = getTime();
	
	string folder = argc > 1 ? argv[1] : "example_extracted_sample";
	string solutionFile = argc > 2 ? argv[2] : "r.csv";
	string featureFile = argc > 3 ? argv[3] : "features.csv";
	string trainFile = argc > 4 ? argv[4] : "traindata.csv";
	string outputFile = argc > 5 ? argv[5] : "rs.csv";
	string outputFile2 = argc > 6 ? argv[6] : "rs2.csv";
	
	if(folder.back() != '/') folder.push_back('/');
	set <string> target = {"radiomics_gtv", "Radiomics_gtv", "radiomics_gtv2", "radiomics_gtv_nw", "radiomics_gtvr"};
	
	string line;
	string linef;
	ifstream in(solutionFile.c_str());
	ifstream in2(featureFile.c_str());
	ofstream out(outputFile.c_str());
	ofstream out2(outputFile2.c_str());
	ofstream out3(trainFile.c_str());
	int ln = 0;
	string prevScanId = "";
	string prevSliceId = "";
	int targetId = 0;
	vector <vector <point_t> > gt;
	while(!in.eof()){
		if(ln % 1000 == 0) clog << ln << "\r";
		getline(in, line);
		getline(in2, linef);
		if(!in.eof()){
			vector <string> row = splitBy(line, ',');
			string scanId = row[0];
			string sliceId = row[1];
			vector <point_t> polygon((row.size() - 2) / 2);
			for(int i = 0; i < polygon.size(); i++){
				polygon[i].x = atof(row[2 * i + 2].c_str());
				polygon[i].y = atof(row[2 * i + 3].c_str());
			}
			if(scanId != prevScanId){
				ifstream inS((folder + scanId + "/structures.dat").c_str());
				string lineS;
				getline(inS, lineS);
				inS.close();
				targetId = 0;
				vector <string> structures = splitBy(lineS, '|');
				for(int j = 0; j < structures.size(); j++){
					if(target.count(structures[j])){
						targetId = j + 1;
						break;
					}
				}
			}
			if(scanId != prevScanId || sliceId != prevSliceId){
				gt.clear();
				if(fileExists(folder + scanId + "/contours/" + sliceId + "." + SSTR(targetId) + ".dat")){
					ifstream inGT((folder + scanId + "/contours/" + sliceId + "." + SSTR(targetId) + ".dat").c_str());
					string lineGT;
					while(!inGT.eof()){
						getline(inGT, lineGT);
						if(!inGT.eof()){
							vector <string> row = splitBy(lineGT, ',');
							gt.push_back(vector <point_t>(row.size() / 3));
							for(int i = 0; i < gt.back().size(); i++){
								gt.back()[i].x = atof(row[3 * i].c_str());
								gt.back()[i].y = atof(row[3 * i + 1].c_str());
							}
						}
					}
					inGT.close();
				}
				prevScanId = scanId;
				prevSliceId = sliceId;
			}
			double a = area(polygon);
			double aGT = 0;
			for(int i = 0; i < gt.size(); i++){
				aGT += fabs(pia_area(&polygon[0], polygon.size(), &gt[i][0], gt[i].size()));
			}
			double ratio = a < 0.000001 ?  0 : aGT / a;
			out << ratio << endl;
			out3 << linef << "," << ratio << endl;
			if(ratio > 0.5) out2 << line << endl;
			ln++;
		}
	}
	in.close();
	in2.close();
	out.close();
	out2.close();
	out3.close();
	clog << ln << " segments found in file." << endl;
	
	clog << getTime() - start << " seconds." << endl;
	
	return 0;
}

