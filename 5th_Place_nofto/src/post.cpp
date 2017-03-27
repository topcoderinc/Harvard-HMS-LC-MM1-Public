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
	
	double QLIM = argc > 1 ? atof(argv[1]) : 0.5;
	double QLIM2 = argc > 2 ? atof(argv[2]) : 0.2;
	string solutionFile = argc > 3 ? argv[3] : "result.csv";
	string outputFile = argc > 4 ? argv[4] : "result_post.csv";
	
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
			/*vector <point_t> & p = polygon[scanId][sliceId].back().second;
			for(int i = 0; i < p.size(); i++){
				p[i].x = atof(row[2 * i + 2].c_str());
				p[i].y = atof(row[2 * i + 3].c_str());
			}*/
			//double a = area(polygon);
			//aGT += fabs(pia_area(&polygon[0], polygon.size(), &gt[i][0], gt[i].size()));
			ln++;
		}
	}
	in.close();
	clog << "    " <<  ln << " polygons found in input file." << endl;
	clog << "    " << polygon.size() << " different scanIDs." << endl;
	
	ofstream out(outputFile.c_str());
	
	int matches = 0;
	for(auto it = polygon.begin(); it != polygon.end(); ++it){
		for(int s = 1; s <= MAXSLICECOUNT; s++){
			vector <Polygon> * slicePolygons = &it->second[s];
			vector <Polygon> * slicePolygonsNear[2] = {&it->second[s - 1], &it->second[s + 1]};
			for(auto pol1 = slicePolygons->begin(); pol1 != slicePolygons->end(); ++pol1){
				bool match = false;
				for(int dir = 0; dir < 2; dir++){
					for(auto pol2 = slicePolygonsNear[dir]->begin(); pol2 != slicePolygonsNear[dir]->end(); pol2++){
						double intersection = fabs(pia_area(&pol1->p[0], pol1->p.size(), &pol2->p[0], pol2->p.size()));
						if((intersection > QLIM * pol1->a/* || intersection > QLIM * pol2->a*/) || (intersection > QLIM2 * pol1->a && intersection > QLIM2 * pol2->a)){
							match = true;
							break;
						}
					}
					if(match) break;
				}
				if(match){
					out << pol1->line << endl;
					matches++;
				}
			}
		}
	}
	clog << "    " << matches << " polygons remains." << endl;
	
	out.close();
	
	
	//clog << getTime() - start << " seconds." << endl;
	
	return 0;
}

