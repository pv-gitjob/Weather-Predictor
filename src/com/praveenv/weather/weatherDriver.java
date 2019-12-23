package com.praveenv.weather;

import weka.classifiers.functions.LinearRegression;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.core.Instance;
import weka.core.Instances;
import java.io.*;
import java.text.SimpleDateFormat;
import java.util.Calendar;

public class weatherDriver {
    public static void main(String[] args) throws Exception {
        String inFile = "Singapore Weather Data.txt";
        Instances dataSet = convertTextToARFF(inFile);
        Instance predictInstance = makePredictionInstance();
        predictWeather(dataSet, predictInstance);
    }

    public static Instance makePredictionInstance() {
        //Create an instance with today's date
        Instance predictInstance = null;
        try {
            String outFile = "Predict.arff";
            FileWriter fstream = new FileWriter(outFile, false);
            BufferedWriter out = new BufferedWriter(fstream);

            //ARRF declarations
            out.write(
                    "@relation prices\n" +
                            "\n" +
                            "@attribute day \t\tdate \"yyyy-MM-dd\"\n" +
                            "@attribute precipitation \treal\n" +
                            "@attribute tempavg \treal\n" +
                            "@attribute temphigh \treal\n" +
                            "@attribute templow \treal\n" +
                            "\n" +
                            "@data"
            );
            //get today's date in weka date format
            SimpleDateFormat simpleDateFormat = new SimpleDateFormat("yyyy-MM-dd");
            String today = simpleDateFormat.format(Calendar.getInstance().getTime());
            out.write(
                    "\n" + '"' + today + '"' + ",?,?,?,?"
            );
            out.close();
            Instances dataSet = new Instances(new BufferedReader(new FileReader("C:\\Users\\Praveen\\Desktop\\Predict.arff")));
            predictInstance = dataSet.lastInstance();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return predictInstance;
    }

    public static void predictWeather(Instances dataSet, Instance predictInstance) {
        //Predict the average weather of predictInstance
        try {
            dataSet.setClassIndex(dataSet.numAttributes()-2);
            predictInstance.setDataset(dataSet);

            //set up linear regression model and get its predictions
            LinearRegression model = new LinearRegression();
            model.buildClassifier(dataSet);
            int temperature = (int) model.classifyInstance(predictInstance);
            System.out.println("Average Weather (Linear Regression): " + temperature);

            //set up multilayer perceptron and get its predictions
            MultilayerPerceptron mlp = new MultilayerPerceptron();
            mlp.setMomentum(0.2);
            mlp.setTrainingTime(100);
            //set hidden layers to automatic
            mlp.setHiddenLayers("a");
            mlp.buildClassifier(dataSet);
            temperature = (int) mlp.classifyInstance(predictInstance);
            System.out.println("Average Weather (Multilayer Perceptron): " + temperature);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static Instances convertTextToARFF(String inFile) {
        //Creates an ARFF file based on the text file and returns the data set created from the ARFF file
        Instances dataSet = null;
        try {
            File file = new File(inFile);
            //set true and comment out the first out.write to just add more data
            String outFile = inFile.substring(0,inFile.lastIndexOf('.')) + ".arff";
            FileWriter fstream = new FileWriter(outFile, false);
            BufferedWriter out = new BufferedWriter(fstream);

            //ARRF declarations
            out.write(
                    "@relation prices\n" +
                            "\n" +
                            "@attribute day \t\tdate \"yyyy-MM-dd\"\n" +
                            "@attribute precipitation \treal\n" +
                            "@attribute tempavg \treal\n" +
                            "@attribute temphigh \treal\n" +
                            "@attribute templow \treal\n" +
                            "\n" +
                            "@data"
            );

            BufferedReader br = null;
            br = new BufferedReader(new FileReader(file));
            String st;
            //read through the title and separator
            br.readLine();
            br.readLine();

            //split lines and use array values according to the data text file
            String[] line;
            st = br.readLine();
            while ((st = br.readLine()) != null) {
                line = st.split("\\s+");//5-date,6-precipitation,7-tempavg,8-temphigh,9-templow
                //the data text lists unknowns values as -9999
                if (line[6].equals("-9999")) {
                    line[6] = "?";
                }
                if (line[7].equals("-9999")) {
                    line[7] = "?";
                }
                if (line[8].equals("-9999")) {
                    line[8] = "?";
                }
                if (line[9].equals("-9999")) {
                    line[9] = "?";
                }
                out.write(
                        "\n" + '"' + line[5].substring(0,4) + '-'  + line[5].substring(4,6)  + '-' + line[5].substring(6,8) + '"'
                                + ',' + line[6] + ',' + line[7] + ',' + line[8] + ',' + line[9]
                );
            }
            out.close();
            dataSet = new Instances(new BufferedReader(new FileReader(outFile)));
        } catch (IOException e) {
            e.printStackTrace();
        }
        return dataSet;
    }

}

