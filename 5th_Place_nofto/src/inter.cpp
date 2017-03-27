#include <bits/stdc++.h>
#include <sys/time.h>
#include <sys/stat.h>
#define SSTR( x ) dynamic_cast< std::ostringstream & >( ( std::ostringstream() << std::dec << x ) ).str()

#include "pia.c"

using namespace std;

const int MAXSLICECOUNT = 500;

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
double area(const vector <point_t>& polygon){
	if(polygon.size() < 3) return 0;
	double a = polygon[0].x * (polygon[1].y - polygon[polygon.size() - 1].y);
	for(int i = 1; i < polygon.size() - 1; i++){
		a += polygon[i].x * (polygon[i+1].y - polygon[i-1].y);
	}
	a += polygon[polygon.size() - 1].x * (polygon[0].y - polygon[polygon.size() - 2].y);
	return fabs(a / 2);
}

struct Polygon{
	string line;
	vector <point_t> p;
	double a;
	Polygon(string & l, vector <string> & row) : line(l){
		p = vector <point_t>((row.size() - 2) / 2);
		for(int i = 0; i < p.size(); i++){
			p[i].x = atof(row[2 * i + 2].c_str());
			p[i].y = atof(row[2 * i + 3].c_str());
		}
		a = area(p);
	}
};

int main(int argc, char* argv[]){
	
	double start = getTime();
	
	string solutionFile = argc > 1 ? argv[1] : "result.csv";
	string outputFile = argc > 2 ? argv[2] : "result_posti.csv";
	
	string line;

	ifstream in(solutionFile.c_str());
	int ln = 0;
	map <string, vector <vector <Polygon> > > polygon;
	while(!in.eof()){
		if(ln % 1000 == 0) clog << ln << "\r";
		getline(in, line);
		if(!in.eof()){
			vector <string> row = splitBy(line, ',');
			string scanId = row[0];
			if(polygon.count(scanId) == 0) polygon[scanId] = vector <vector <Polygon> >(MAXSLICECOUNT+2);
			int sliceId = atoi(row[1].c_str());
			polygon[scanId][sliceId].push_back(Polygon(line, row));
			ln++;
		}
	}
	in.close();
	clog << "    " << ln << " polygons found in input file." << endl;
	clog << "    " << polygon.size() << " different scanIDs." << endl;
	
	ofstream out(outputFile.c_str());
	
	int matches = 0;
	int smallmatches = 0;
	for(auto it = polygon.begin(); it != polygon.end(); ++it){
		for(int s = 1; s <= MAXSLICECOUNT; s++){
			vector <Polygon> & sp = it->second[s];
			for(int i = 0; i < sp.size(); i++){
				bool match = false;
				for(int j = 0; j < sp.size(); j++) if(i != j){
					double intersection = fabs(pia_area(&sp[i].p[0], sp[i].p.size(), &sp[j].p[0], sp[j].p.size()));
					if((intersection >= 0.99*sp[i].a)){
						match = true;
						break;
					}
					if((intersection >= 0.05*sp[i].a)){
						smallmatches++;
						//out << intersection / sp[i].a << endl;
						//out << sp[i].line << endl << sp[j].line << endl << endl;
					}
				}
				if(match){
					matches++;
				}
				else out << sp[i].line << endl;
			}
		}
	}
	clog << "    " << matches << " intersections detected and removed." << endl;
	clog << "    " << smallmatches << " small intersections detected and not removed." << endl;
	
	out.close();
	
	
	//clog << getTime() - start << " seconds." << endl;
	
	return 0;
}

