package com.ggstar.serving.jpmml.load;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.Evaluator;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.InputField;
import org.jpmml.evaluator.ModelEvaluatorFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class JavaModelServer {

    String modelPath;

    Evaluator model;

    public JavaModelServer(String modelPath){
        this.modelPath = modelPath;
    }

    public void loadModel(){
        PMML pmml;
        try(InputStream is = new FileInputStream(new File(modelPath))){
            pmml = org.jpmml.model.PMMLUtil.unmarshal(is);
            ModelEvaluatorFactory modelEvaluatorFactory = ModelEvaluatorFactory.newInstance();
            this.model = modelEvaluatorFactory.newModelEvaluator(pmml);
            this.model.verify();
            /*List<InputField> inputFields = this.model.getInputFields();
            for (InputField inputField : inputFields){
                System.out.println(inputField.getName().getValue());
            }*/
        }catch (Exception e){
            System.err.println(e);
        }
    }

    public Map<FieldName, ?> forecast(Map<String, ?> featureMap){
        if (this.model == null){
            loadModel();
        }
        if (featureMap == null){
            System.err.println("features is null");
            return null;
        }

        List<InputField> inputFields = this.model.getInputFields();
        Map<FieldName, FieldValue> pmmlFeatureMap = new LinkedHashMap<>();
        for (InputField inputField : inputFields){
            if (featureMap.containsKey(inputField.getName().getValue())) {
                Object value = featureMap.get(inputField.getName().getValue());
                pmmlFeatureMap.put(inputField.getName(), inputField.prepare(value));
            }else{
                System.err.println("lack of feature: " + inputField.getName().getValue());
                return null;
            }
        }
        return this.model.evaluate(pmmlFeatureMap);
    }
}
