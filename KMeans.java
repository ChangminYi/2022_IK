
/**
 * 20170454 Yi Changmin
 * Information and Knowledge k-Means Clustering project
 */

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.StringTokenizer;

class Data {
    int cluster;
    public double[] param = new double[KMeans.DATA_DIM];

    Data() {
        Arrays.fill(param, 0);
    }

    /**
     * @param str : raw string read from .csv
     */
    Data(final String str) {
        StringTokenizer parser = new StringTokenizer(str, ",");
        for (int i = 0; i < KMeans.DATA_DIM; i++) {
            param[i] = Double.parseDouble(parser.nextToken());
        }
    }

    /**
     * @param lhs left side operand
     * @param rhs right side operand
     * @return euclid distance of lhs, rhs
     */
    public static double getDistance(final Data lhs, final Data rhs) {
        double ret = 0;
        for (int i = 0; i < lhs.param.length; i++) {
            ret += (lhs.param[i] - rhs.param[i]) * (lhs.param[i] - rhs.param[i]);
        }
        return Math.sqrt(ret);
    }

    /**
     * @param other other Data class to compare
     * @return sameness of param of two Data
     */
    public boolean isSame(final Data other) {
        for (int i = 0; i < this.param.length; i++) {
            if (this.param[i] != other.param[i]) {
                return false;
            }
        }
        return true;
    }

    /**
     * copies param values from other Data class
     * @param other original data source
     */
    public void copyFrom(final Data other) {
        for(int i = 0; i < KMeans.DATA_DIM; i++) {
            this.param[i] = other.param[i];
        }
    }
}

public class KMeans {
    final private static String outputFileName = "20170454.csv";
    final public static int DATA_DIM = 10;

    private static int K_VALUE;
    private static File inputFile, outputFile;
    private static BufferedReader reader;
    private static BufferedWriter writer;

    private static ArrayList<Data> data = new ArrayList<>();
    private static Data[] center;
    private static double[] minValue = new double[DATA_DIM];
    private static double[] maxValue = new double[DATA_DIM];

    /**
     * @param args[0] input data file name
     * @param args[1] k value
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        if (args.length != 2) { // checking argument errors
            System.out.println("Parameter format is wrong");
            return;
        } else {
            inputFile = new File(args[0]);
            if (!inputFile.exists()) {
                System.out.println("There's no file named " + args[0]);
                return;
            }

            try {
                K_VALUE = Integer.parseInt(args[1]);
            } catch (NumberFormatException e) {
                System.out.println("Not correct number format");
                return;
            }
            if(K_VALUE < 1) {
                System.out.println("Invalid K value. K must bigger than 0");
                return;
            }
        }

        // read .csv file and convert to Data class
        reader = new BufferedReader(new FileReader(inputFile));
        for (String line; (line = reader.readLine()) != null;) {
            data.add(new Data(line));
        }
        reader.close();

        getMinAndMax();
        normalizeData(); // MinMax normalization

        int loopCount = 1;
        long startTime = System.nanoTime();
        initializeCluster();
        while (true) { // do K-means clustering while there's change in class
            center = getNewCenter();
            if (clusterData()) {
                loopCount++;
                continue;
            } else {
                break;
            }
        }
        long endTime = System.nanoTime();

        outputFile = new File(outputFileName); // writing result to output file
        if (!outputFile.exists())
            outputFile.createNewFile();
        writer = new BufferedWriter(new FileWriter(outputFile));
        for (int i = 0; i < data.size(); i++) {
            for (int j = 0; j < data.size(); j++) {
                writer.write(
                        (data.get(i).cluster == data.get(j).cluster ? "1" : "0") + (j < data.size() - 1 ? "," : "\n")
                );
            }
        }
        writer.flush();
        writer.close();

        // printing execution time
        double timeInMsec = ((double)endTime - startTime) / 1000000;
        System.out.printf("Runtime (Only clustering): %.3f msec\n", timeInMsec);
        System.out.printf("Mean of time spend in one iteration: %.3f msec\n", timeInMsec / loopCount);
        
        // printing cluster distance sum and size
        int[] clusterSize = getClusterSize();
        double[] totalDistance = getTotalDistance();
        double totalDistanceSum = 0;
        System.out.println("Distance sum of each cluster and size (Normalized scale):");
        for (int i = 0; i < K_VALUE; i++) {
            totalDistanceSum += totalDistance[i];
            System.out.printf("Cluster %d: %.3f, size = %d\n", i, totalDistance[i], clusterSize[i]);
        }
        System.out.printf("Total sum of distance: %.3f\n", totalDistanceSum);

        // printing coordinate of cluster center
        denormalizeCenter();
        System.out.println("Coordinate of center (Original scale): ");
        for (int i = 0; i < K_VALUE; i++) {
            System.out.print("Center " + i + ": ");
            for (int j = 0; j < DATA_DIM; j++) {
                System.out.printf("%.3f ", center[i].param[j]);
            }
            System.out.println();
        }
    }

    /**
     * assign cluster to each data by using cyclic method
     */
    private static void initializeCluster() {
        for (int i = 0; i < data.size(); i++) {
            data.get(i).cluster = i % K_VALUE;
        }
    }

    /**
     * @return array of new centers, which are derived from previous classifyData()
     */
    private static Data[] getNewCenter() {
        int[] clusterCnt = new int[K_VALUE];
        Data[] newCenter = new Data[K_VALUE];

        Arrays.fill(clusterCnt, 0);
        for (int i = 0; i < K_VALUE; i++) {
            newCenter[i] = new Data();
        }
        for (int i = 0; i < data.size(); i++) {
            clusterCnt[data.get(i).cluster]++;
            for (int j = 0; j < DATA_DIM; j++) {
                newCenter[data.get(i).cluster].param[j] += data.get(i).param[j];
            }
        }

        // if no object is in cluster, assign new center (randomly in data)
        for (int i = 0; i < K_VALUE; i++) {
            if (clusterCnt[i] == 0) {
                Data randomCenter = data.get((int) (Math.random() * data.size()));
                while(true) {
                    boolean hasSame = false;
                    for(int j = 0; j < K_VALUE; j++) {
                        if(clusterCnt[j] > 0 && newCenter[j].isSame(randomCenter)) {
                            hasSame = true;
                            break;
                        }
                    }

                    if(hasSame)
                        randomCenter = data.get((int) (Math.random() * data.size()));
                    else
                        break;
                }

                newCenter[i].copyFrom(randomCenter);
                clusterCnt[i] = 1;
            }
        }

        for (int i = 0; i < K_VALUE; i++) {
            for (int j = 0; j < DATA_DIM; j++) {
                newCenter[i].param[j] /= clusterCnt[i];
            }
        }

        return newCenter;
    }

    /**
     * assign new cluster to each data
     * @return whether any of data has changed its class
     */
    private static boolean clusterData() {
        boolean hasChanged = false;

        for (int i = 0; i < data.size(); i++) {
            int newCluster = -1;
            double minDistance = Double.MAX_VALUE;

            for (int j = 0; j < K_VALUE; j++) {
                double tmpDistance = Data.getDistance(data.get(i), center[j]);
                if (tmpDistance < minDistance) {
                    newCluster = j;
                    minDistance = tmpDistance;
                }
            }

            if (newCluster != data.get(i).cluster) {
                hasChanged = true;
                data.get(i).cluster = newCluster;
            }
        }

        return hasChanged;
    }

    /**
     * MinMax normalization. convert data to normalized scale
     */
    private static void normalizeData() {
        for(Data cur : data) {
            for(int i = 0; i < DATA_DIM; i++) {
                cur.param[i] = (cur.param[i] - minValue[i]) / (maxValue[i] - minValue[i]);
            }
        }
    }

    /**
     * get min and max value of each data column
     */
    private static void getMinAndMax() {
        Arrays.fill(minValue, Double.MAX_VALUE);
        Arrays.fill(maxValue, -Double.MAX_VALUE);

        for(final Data cur : data) {
            for(int i = 0; i < DATA_DIM; i++) {
                minValue[i] = Math.min(minValue[i], cur.param[i]);
                maxValue[i] = Math.max(maxValue[i], cur.param[i]);
            }
        }
    }

    /**
     * convert normalized scale to original scale
     */
    private static void denormalizeCenter() {
        for(Data cur : center) {
            for(int i = 0; i < DATA_DIM; i++) {
                cur.param[i] = cur.param[i] * (maxValue[i] - minValue[i]) + minValue[i];
            }
        }
    }

    /**
     * @return distance sum of each cluster
     */
    private static double[] getTotalDistance() {
        double[] totalDistance = new double[K_VALUE];
        Arrays.fill(totalDistance, 0);

        for (final Data cur : data) {
            totalDistance[cur.cluster] += Data.getDistance(cur, center[cur.cluster]);
        }

        return totalDistance;
    }

    /**
     * @return size of each cluster
     */
    private static int[] getClusterSize() {
        int[] sizeCount = new int[K_VALUE];
        Arrays.fill(sizeCount, 0);

        for(final Data cur : data) {
            sizeCount[cur.cluster]++;
        }

        return sizeCount;
    }
}
