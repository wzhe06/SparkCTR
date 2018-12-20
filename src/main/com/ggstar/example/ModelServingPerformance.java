package com.ggstar.example;

import com.ggstar.ctrmodel.InnerProductNNCtrModel;
import com.ggstar.features.FeatureEngineering;
import com.ggstar.serving.jpmml.load.JavaModelServer;
import com.ggstar.util.Scala2JavaConverter;
import ml.combust.mleap.core.types.StructField;
import ml.combust.mleap.core.types.StructType;
import ml.combust.mleap.runtime.javadsl.LeapFrameBuilder;
import ml.combust.mleap.tensor.DenseTensor;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.ml.linalg.DenseVector;
import org.apache.spark.sql.*;
import org.dmg.pmml.FieldName;

import java.util.*;

public class ModelServingPerformance {


    public static void main1(String[] args){

        Logger.getLogger("org").setLevel(Level.ERROR);

        SparkConf conf = new SparkConf()
                .setMaster("local")
                .setAppName("ctrModel")
                .set("spark.submit.deployMode", "client");

        SparkSession spark = SparkSession.builder().config(conf).appName("ctrModel").getOrCreate();

        Dataset<Row> rawSamples = spark.read().format("orc").option("compression", "snappy").load("data/samples.snappy.orc");

        //transform array to vector for following vectorAssembler
        Dataset<Row> samples = FeatureEngineering.transferArray2Vector(rawSamples);

        //model training
        System.out.println("Train Neural Network Ctr Prediction Model:");
        InnerProductNNCtrModel innModel = new InnerProductNNCtrModel();
        innModel.train(samples);
        Dataset<Row> transformedData = innModel.transform(samples);


        //model serialization by mleap
        com.ggstar.serving.mleap.serialization.ModelSerializer mleapModelSerializer = new com.ggstar.serving.mleap.serialization.ModelSerializer();
        //mleapModelSerializer.serializeModel(innModel._pipelineModel(), "jar:file:/Users/zhwang/Workspace/CTRmodel/model/inn.model.mleap.zip", transformedData);

        //model serialization by JPMML
        com.ggstar.serving.jpmml.serialization.ModelSerializer jpmmlModelSerializer = new com.ggstar.serving.jpmml.serialization.ModelSerializer();
        //jpmmlModelSerializer.serializeModel(innModel._pipelineModel(), "model/inn.model.jpmml.xml", transformedData);

        //inference time by original spark model
        Encoder<Sample> sampleEncoder = Encoders.bean(Sample.class);

        int user_id = 20143;
        int item_id = 52;
        int category_id = 16;
        String content_type = "movie";
        String timestamp = "1533487890";
        long user_item_click = 0;
        double user_item_imp = 0.69314718;
        double item_ctr = 0.117256d;
        int is_new_user = 1;
        double embedding_inner_product = 0.5;

        int testRound = 5000;

        List<Double> ctrList = new LinkedList<>();

        long startTimestamp = System.currentTimeMillis();
        for (int i = 0 ; i < testRound; i++) {

            Sample sample = new Sample(user_id, item_id, category_id, content_type, timestamp, user_item_click, user_item_imp, item_ctr *= 1.00000011, is_new_user, embedding_inner_product);
            ArrayList<Sample> samplesList = new ArrayList<>();
            samplesList.add(sample);
            Dataset<Sample> sampleDataset = spark.createDataset(
                    samplesList,
                    sampleEncoder
            );

            //sampleDataset.show(1, false);

            Dataset<org.apache.spark.sql.Row> result = innModel._pipelineModel().transform(sampleDataset);
            ctrList.add(((DenseVector)result.head().getAs("probability")).apply(1));
        }

        System.out.println(System.currentTimeMillis() - startTimestamp);


        /*
        result.printSchema();
        System.out.println(((DenseVector)result.head().getAs("rawPrediction")).apply(0));
        System.out.println(((DenseVector)result.head().getAs("rawPrediction")).apply(1));
        System.out.println(((DenseVector)result.head().getAs("probability")).apply(0));
        System.out.println(((DenseVector)result.head().getAs("probability")).apply(1));
           */
        //inference by mleap model

        LeapFrameBuilder builder = new LeapFrameBuilder();
        List<StructField> fields = new ArrayList();
        fields.add(builder.createField("user_id", builder.createInt()));
        fields.add(builder.createField("item_id", builder.createInt()));
        fields.add(builder.createField("category_id", builder.createInt()));
        fields.add(builder.createField("content_type", builder.createString()));
        fields.add(builder.createField("timestamp", builder.createString()));
        fields.add(builder.createField("user_item_click", builder.createLong()));
        fields.add(builder.createField("user_item_imp", builder.createDouble()));
        fields.add(builder.createField("item_ctr", builder.createDouble() ));
        fields.add(builder.createField("is_new_user", builder.createInt()));
        fields.add(builder.createField("embedding_inner_product", builder.createDouble()));
        StructType schema = builder.createSchema(fields);
        com.ggstar.serving.mleap.load.JavaModelServer mleapModelServer = new com.ggstar.serving.mleap.load.JavaModelServer("model/inn.model.mleap.zip", schema);



        startTimestamp = System.currentTimeMillis();
        for (int i = 0 ; i < testRound; i++) {

            ml.combust.mleap.runtime.frame.Row features = builder.createRow(user_id, item_id, category_id, content_type, timestamp, user_item_click, user_item_imp, item_ctr *= 1.00000011, is_new_user, embedding_inner_product);
            ml.combust.mleap.runtime.frame.Row mleapResult = mleapModelServer.forecast(features);
            ctrList.add(Scala2JavaConverter.pauseCtr(mleapResult.getTensor(16).toDense()));
        }
        System.out.println(System.currentTimeMillis() - startTimestamp);

        //DenseTensor prob = mleapResult.getTensor(16).toDense();
        //System.out.println("ctr of mleap:\t" + Scala2JavaConverter.pauseCtr(prob));


        //inference by JPMML model
        JavaModelServer jpmmlModelServer = new JavaModelServer("model/inn.model.jpmml.xml");

        startTimestamp = System.currentTimeMillis();

        for (int i = 0 ; i < testRound; i++) {

            HashMap<String, Object> featureMap = new HashMap<>();
            featureMap.put("user_id", user_id);
            featureMap.put("item_id", item_id);
            featureMap.put("category_id", category_id);
            featureMap.put("content_type", content_type);
            featureMap.put("timestamp", timestamp);
            featureMap.put("user_item_click", user_item_click);
            featureMap.put("user_item_imp", user_item_imp);
            featureMap.put("item_ctr", item_ctr *= 1.00000011);
            featureMap.put("is_new_user", is_new_user);
            featureMap.put("embedding_inner_product", embedding_inner_product);

            Map<FieldName, ?> jpmmlResult = jpmmlModelServer.forecast(featureMap);
            ctrList.add((Double)jpmmlResult.get(new FieldName("probability(1)")));
        }

        System.out.println(System.currentTimeMillis() - startTimestamp);
        //System.out.println("ctr of jpmml:\t" + jpmmlResult.get(new FieldName("probability(1)")));
    }

    public static void main(String[] args){
        int user_id = 20143;
        int item_id = 52;
        int category_id = 16;
        String content_type = "movie";
        String timestamp = "1533487890";
        long user_item_click = 0;
        double user_item_imp = 0.69314718;
        double item_ctr = 0.117256d;
        int is_new_user = 0;
        double embedding_inner_product = 0.5;

        int testRound = 10000;

        JavaModelServer jpmmlModelServer = new JavaModelServer("model/inn.model.jpmml.xml");

        long startTimestamp = System.currentTimeMillis();
        List<Double> ctrList = new LinkedList<>();

        for (int i = 0 ; i < testRound; i++) {
            user_item_imp *= 1.001;

            HashMap<String, Object> featureMap = new HashMap<>();
            featureMap.put("user_id", user_id);
            featureMap.put("item_id", item_id);
            featureMap.put("category_id", category_id);
            featureMap.put("content_type", content_type);
            featureMap.put("timestamp", timestamp);
            featureMap.put("user_item_click", user_item_click);
            featureMap.put("user_item_imp", user_item_imp);
            featureMap.put("item_ctr", item_ctr);
            featureMap.put("is_new_user", is_new_user);
            featureMap.put("embedding_inner_product", embedding_inner_product);

            Map<FieldName, ?> jpmmlResult = jpmmlModelServer.forecast(featureMap);
            ctrList.add((Double)jpmmlResult.get(new FieldName("probability(1)")));
        }

        System.out.println(System.currentTimeMillis() - startTimestamp);

        for (double ctr : ctrList) {
            System.out.println(ctr);
        }
    }
}
