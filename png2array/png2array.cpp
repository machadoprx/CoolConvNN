#include "png2array.h"

namespace fs = std::filesystem;
std::map<int, std::string> labelNames;

double *decodeTwoSteps(std::string filename, int &w, int &h) {
    
    std::vector<unsigned char> png;
    std::vector<unsigned char> image; //the raw pixels
    unsigned width, height;

    //load and decode
    unsigned error = lodepng::load_file(png, filename);
    if(!error) error = lodepng::decode(image, width, height, png);

    //if there's an error, display it
    if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

    double *pixels = new double[width * height];
    w = width, h = height;

    for (int i = 0; i < (int)(width * height); i++) {
        int r = image.at((i * 4) + 0);
        int g = image.at((i * 4) + 1);
        int b = image.at((i * 4) + 2);
        double gray = ((r * 0.299) + (g * 0.587) + (b * 0.114));
        pixels[i] = gray;
    }

    return pixels;
}

void normalizeData(double** &raw, double* &mean, double* &deviation, int samples, int featureSize) {

    double e = 0.0000001;
    
    mean = new double[featureSize];
    deviation = new double[featureSize];

    for (int j = 0; j < featureSize; j++) {

        double featureSum = 0;

        for (int i = 0; i < samples; i++) {
            featureSum += raw[i][j];
        }

        mean[j] = featureSum / samples;

        // mean subtraction
        double featureVariance = 0;

        for (int i = 0; i < samples; i++) {

            raw[i][j] = raw[i][j] - mean[j];
            featureVariance += (raw[i][j] * raw[i][j]);
            
        }

        featureVariance = featureVariance / samples;

        // divide by standard deviation
        deviation[j] = sqrt(featureVariance + e);

        for (int i = 0; i < samples; i++) {
            raw[i][j] = raw[i][j] / deviation[j];
        }
    }
}

double **png2data(std::string dataPath, int labels, int samplesPerLabel) {

    int index = 0;
    int label = 0;
    double **raw = new double*[labels * samplesPerLabel];

    for (const auto & entry : fs::directory_iterator(dataPath)){
 
        if (entry.is_directory()) {
            
            std::string labelPath = entry.path().string();

            labelNames[label] = entry.path().filename().string();
            bool print = true;
            
            for (const auto & sample : fs::directory_iterator(labelPath)) {

                int w, h;
                raw[index] = decodeTwoSteps(sample.path().string(), w, h);
                index++;
                if ((index * 100 / (samplesPerLabel * labels)) % 5 == 0 && print) {
                    std::cout << index * 100 / (samplesPerLabel * labels) << "%\n";
                    print = false;
                }
        
            }
            std::cout << index << " " << entry.path().filename().string() << '\n';
            label++;
        }
    }

    std::cout << "Found " << label<< " classes\n";

    return raw;
}

void saveData(const char* outputPath, double** &data, double* &mean, double* &deviation, int labels, int samplesPerLabel, int featuresSize) {

    FILE *f = fopen(outputPath, "wb");
    
    fwrite(&labels, sizeof(int), 1, f);
    fwrite(&samplesPerLabel, sizeof(int), 1, f);
    fwrite(&featuresSize, sizeof(int), 1, f);

    fwrite(mean, sizeof(double) * featuresSize, 1, f);
    fwrite(deviation, sizeof(double) * featuresSize, 1, f);

    for (int i = 0; i < labels * samplesPerLabel; i++) {
        fwrite(data[i], sizeof(double) * featuresSize, 1, f);
    }

    fclose(f);
}

void saveLabelNames(std::string namesOut) {

    std::ofstream out(namesOut);

    for (int i = 0; i < (int)labelNames.size(); i++) {
        out << i << " " << labelNames[i] << std::endl;
    }

    out.close();
}

void processData(const char *dataPath, int samplesPerLabel, int size, int labels, const char *outPath) {

    double **rawData = png2data(dataPath, labels, samplesPerLabel);

    double *mean = nullptr;
    double *deviation = nullptr;
    normalizeData(rawData, mean, deviation, labels * samplesPerLabel, size * size);

    saveData(outPath, rawData, mean, deviation, labels, samplesPerLabel, size * size);

    std::string namesOut = std::string(outPath);
    namesOut += ".names";
    saveLabelNames(namesOut);

    for (int i = 0; i < samplesPerLabel * labels; i++)
        delete[] rawData[i];

    delete[] rawData;
    delete[] mean;
    delete[] deviation;

}

void loadData(const char* path, double** &data, double* &mean, double* &deviation, int &labels, int &samplesPerLabels, int &featuresDimension) {

    FILE *fp = fopen(path, "rb");

    fread(&labels, sizeof(int), 1, fp);
    fread(&samplesPerLabels, sizeof(int), 1, fp);
    fread(&featuresDimension, sizeof(int), 1, fp);

    mean = new double[featuresDimension];
    deviation = new double[featuresDimension];
    data = new double*[labels * samplesPerLabels];

    for (int i = 0; i < labels * samplesPerLabels; i++) {
        data[i] = new double[featuresDimension];
    }

    fread(mean, sizeof(double) * featuresDimension, 1, fp);
    fread(deviation, sizeof(double) * featuresDimension, 1, fp);

    for (int i = 0; i < labels * samplesPerLabels; i++) {
        fread(data[i], sizeof(double) * featuresDimension, 1, fp);
    }

    fclose(fp);
}

int *genTargets(int labels, int samplesPerLabels) {

    auto targets = new int[labels * samplesPerLabels];

    for (int i = 0; i < labels; i++) {
        for (int j = 0; j < samplesPerLabels; j++) {
            targets[i * samplesPerLabels + j] = i;
        }
    }

    return targets;
}
