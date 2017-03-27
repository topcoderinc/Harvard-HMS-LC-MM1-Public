#include <bits/stdc++.h>
#include <sys/time.h>
#include <sys/stat.h>
#define SSTR( x ) dynamic_cast< std::ostringstream & >( ( std::ostringstream() << std::dec << x ) ).str()

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

int main(int argc, char* argv[]){
	
	double start = getTime();
	
	double threshold = argc > 1 ? atof(argv[1]) : 0.1;
	string solutionFile = argc > 2 ? argv[2] : "result_oob.csv";
	string outputFile = argc > 3 ? argv[3] : "result_oob_selected.csv";
	
	string line;
	ifstream in(solutionFile.c_str());
	ofstream out(outputFile.c_str());
	int outCount = 0;
	int ln = 0;
	while(!in.eof()){
		if(ln % 10000 == 0) clog << ln << "\r";
		getline(in, line);
		if(!in.eof()){
			string number = "";
			int c = 0;
			while(line[c] != ','){
				number.push_back(line[c]);
				c++;
			}
			double r = atof(number.c_str());
			if(r > threshold){
				out << line.substr(c + 1) << endl;
				outCount++;
			}
			ln++;
		}
	}
	in.close();
	out.close();
	clog << ln << " segments found in file." << endl;
	clog << outCount << " segments with result > threshold." << endl;
	
	clog << getTime() - start << " seconds." << endl;
	
	return 0;
}

