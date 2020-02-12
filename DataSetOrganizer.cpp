//
// Created by vmachado on 2/12/20.
//

#include "DataSetOrganizer.h"

/*DataSetOrganizer::DataSetOrganizer(const char *path, int width, int height, bool histogram) {

    File f = new File(folder);
    try {
        assert f.isDirectory();
    }
    catch (Exception e){
        System.err.println(e.toString());
        return;
    }
    this.featuresDimension = width * height;
    this.dataSetPath = folder;
    this.dataSetWidth = width;
    this.dataSetHeight = height;
    this.histogram = histogram;
}

private ArrayList<int[]> extractHistogram(File labelDirectory) {

    ArrayList<int[]> labelHistograms = new ArrayList<>();

    for (File imgFile: Objects.requireNonNull(labelDirectory.listFiles())){

        BufferedImage img;

        try{
            img = ImageIO.read(imgFile);
            assert img.getWidth() == dataSetWidth && img.getHeight() == dataSetHeight;
        }
        catch (IOException e){
            System.err.println(imgFile.getName() + " corrupted");
            continue;
        }
        catch (AssertionError e){
            System.err.println(imgFile.getName() + " incorrect size");
            continue;
        }

        int[] aFaceFeaturesHist = new LocalBinaryPattern(img).getLBPHistogram();
        labelHistograms.add(aFaceFeaturesHist);
        samples++;
    }
    return labelHistograms;
}

public static int[] getPixels(BufferedImage img, int width, int height) {

    int[] pixels = new int[width * height];
    int index = 0;

    for (int x = 0; x < width; x++) {
        for (int y = 0; y < height; y++) {

            int rgb = img.getRGB(x, y);
            int r = (rgb >> 16) & 0xff;
            int g = (rgb >> 8) & 0xff;
            int b = (rgb & 0xff);

            pixels[index] = (int)((r * 0.299) + (g * 0.587) + (b * 0.114));

            index++;
        }
    }

    return pixels;
}

private ArrayList<int[]> extractPixels(File labelDirectory) {

    ArrayList<int[]> labelData = new ArrayList<>();

    for (File imgFile: Objects.requireNonNull(labelDirectory.listFiles())){

        BufferedImage img;

        try{
            img = ImageIO.read(imgFile);
            assert img.getWidth() == dataSetWidth && img.getHeight() == dataSetHeight;
        }
        catch (IOException e){
            System.err.println(imgFile.getName() + " corrupted");
            continue;
        }
        catch (AssertionError e){
            System.err.println(imgFile.getName() + " incorrect size");
            continue;
        }
        int[] sampleData = getPixels(img, dataSetWidth, dataSetHeight);
        labelData.add(sampleData);
        samples++;
    }
    return labelData;
}

public void organizeData() {

    File dataSetFolder = new File(dataSetPath);
    String[] dataSetLabels = dataSetFolder.list();
    assert dataSetLabels != null;

    labelsSize = 0;
    samples = 0;
    String separator;

    if(System.getProperty("os.name").equals("Linux")) separator = "/";
    else separator = "\\";

    for (String label : dataSetLabels) {

        final String labelDirString = dataSetPath + separator + label;
        final File labelDir = new File(labelDirString);

        if (labelDir.isDirectory()){

            labelsNames.put(labelsSize, label);
            ArrayList<int[]> labelData;

            if (histogram) {
                labelData = extractHistogram(labelDir);
            }
            else {
                labelData = extractPixels(labelDir);
            }

            dataSet.put(labelsSize, labelData);
            System.out.println(label + " computation completed");
            labelsSize++;
        }

        else{
            System.err.println(labelDir + " not a folder");
        }
    }

    organized = true;
}

private double[][] processData(int[][] data, double[] featuresMean, double[] featuresSTD) {

    double[][] res = new double[data[0].length][data.length];

    int[][] dataT = transposedInt(data);
    double e = 0.0000001;

    for (int i = 0; i < data[0].length; i++) {

        double featureSum = 0;

        for (int j = 0; j < data.length; j++) {
            featureSum += dataT[i][j];
        }

        featuresMean[i] = featureSum / (double)data.length;

        // mean subtraction
        double featureVariance = 0;

        for (int j = 0; j < data.length; j++) {
            res[i][j] = dataT[i][j] - featuresMean[i];
            featureVariance += (res[i][j] * res[i][j]);
        }

        featureVariance = featureVariance / (double) data.length;

        // divide by standard deviation
        featuresSTD[i] = Math.sqrt(featureVariance + e);

        for (int j = 0; j < data.length; j++) {
            res[i][j] = res[i][j] / featuresSTD[i];
        }
    }

    return transposed(res);
}

private int[][] getInputMatrix(){

    if(!organized){
        return null;
    }

    int[][] input = new int[samples][featuresDimension];
    int inputIndex = 0;

    for (int i = 0; i < labelsSize; i++) {

        ArrayList<int[]> labelData = dataSet.get(i);

        for (int[] data : labelData) {
            input[inputIndex] = data.clone();
            inputIndex++;
        }
    }

    return input;
}

private int[] getOutputMatrix(){

    if(!organized){
        return null;
    }

    int[] output = new int[samples];
    int outputIndex = 0;

    for (int i = 0; i < labelsSize; i++) {

        ArrayList<int[]> labelData = dataSet.get(i);

        for (int j = 0; j < labelData.size(); j++) {
            output[outputIndex] = i;
            outputIndex++;
        }
    }
    return output;
}

public void saveData(String path) throws IOException {

        if(!organized){
            return;
        }

        int[][] inputMatrix = getInputMatrix();

        assert inputMatrix != null;

        double[] featuresMean = new double[featuresDimension];
        double[] featuresSTD = new double[featuresDimension];
        double[][] processedMatrix = processData(inputMatrix, featuresMean, featuresSTD);

        FileOutputStream fileOut = new FileOutputStream(path);
        ObjectOutputStream out = new ObjectOutputStream(fileOut);
        out.writeObject(processedMatrix);
        out.writeObject(featuresMean);
        out.writeObject(featuresSTD);
        out.writeObject(getOutputMatrix());
        out.close();
        fileOut.close();
}

static double[][] transposed(double[][] W) {

    double[][] res = new double[W[0].length][W.length];

    IntStream.range(0, W.length).parallel().forEach(i -> {
        for (int j = 0; j < W[0].length; j++) {
            res[j][i] = W[i][j];
        }
    });

    return res;
}

static int[][] transposedInt(int[][] W) {

    int[][] res = new int[W[0].length][W.length];

    IntStream.range(0, W.length).parallel().forEach(i -> {
        for (int j = 0; j < W[0].length; j++) {
            res[j][i] = W[i][j];
        }
    });

    return res;
}

public HashMap<Integer, String> getLabelsNames(){
    return this.labelsNames;
}*/