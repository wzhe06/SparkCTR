package com.ggstar.example;

import com.ggstar.serving.jpmml.load.JavaModelServer;
import org.dmg.pmml.FieldName;

import java.util.HashMap;
import java.util.Map;

public class JPMMLModelServing {
    public static void main(String[] args){

        JavaModelServer javaModelServer = new JavaModelServer("model/inn.model.jpmml.xml");

        HashMap<String, Object> featureMap = new HashMap<>();

        featureMap.put("user_id", 20143);
        featureMap.put("item_id", 52);
        featureMap.put("category_id", 16);
        featureMap.put("content_type", "movie");
        featureMap.put("timestamp", "1533487890");
        featureMap.put("user_item_click", 0L);
        featureMap.put("user_item_imp", 0.69314718d);
        featureMap.put("item_ctr", 0.00617256d);
        featureMap.put("is_new_user", 0);
        featureMap.put("embedding_inner_product", 0.5);

        Map<FieldName, ?> result = javaModelServer.forecast(featureMap);

        for (Map.Entry<FieldName, ?> field : result.entrySet()){
            System.out.println(field.getKey().getValue() + ":\t" +  field.getValue());
        }

        for(int i = 0 ; i < result.size(); i++){
            System.out.println(result);
        }

        System.out.println(result.get(new FieldName("probability(1)")));

    }
}
