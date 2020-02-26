#include <vector>
#include <iostream>
#include <string>
#include <filesystem>
#include <map>
#include <cmath>
#include <fstream>
#include "lodepng.h"

void processData(const char *dataPath, int samplesPerLabel, int size, int labels, const char *outPath);
void loadData(const char* path, float** &data, float* &mean, float* &deviation, int &labels, int &samplesPerLabels, int &featuresDimension);
int *genTargets(int labels, int samplesPerLabels);
float *decodeTwoSteps(std::string filename, int &w, int &h);