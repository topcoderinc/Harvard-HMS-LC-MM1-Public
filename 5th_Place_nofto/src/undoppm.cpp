#include <bits/stdc++.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <dirent.h>

using namespace std;


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
	
	if(folder.back() != '/') folder.push_back('/');
	vector <string> scans = listdirdir(folder);
	
	for(int i = 0; i < scans.size(); i++){
		system(("rm -rf " + folder + scans[i] + "/ppm").c_str());
		clog << "       \r";
	}
	
	return 0;
}

