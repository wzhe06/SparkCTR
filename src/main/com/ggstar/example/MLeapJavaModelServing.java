package com.ggstar.example;

import com.ggstar.serving.mleap.load.JavaModelServer;
import com.ggstar.util.Scala2JavaConverter;
import ml.combust.mleap.core.types.StructField;
import ml.combust.mleap.core.types.StructType;
import ml.combust.mleap.runtime.frame.Row;
import ml.combust.mleap.runtime.javadsl.LeapFrameBuilder;
import ml.combust.mleap.tensor.DenseTensor;

import java.util.ArrayList;
import java.util.List;

public class MLeapJavaModelServing {
    public static void main(String[] args){
        LeapFrameBuilder builder = new LeapFrameBuilder();
        List<StructField> fields = new ArrayList();
        fields.add(builder.createField("user_id", builder.createInt()));
        fields.add(builder.createField("item_id", builder.createInt()));
        fields.add(builder.createField("category_id", builder.createInt()));
        fields.add(builder.createField("content_type", builder.createString()));
        fields.add(builder.createField("timestamp", builder.createString()));
        fields.add(builder.createField("user_item_click", builder.createLong()));
        fields.add(builder.createField("user_item_imp", builder.createDouble()));
        fields.add(builder.createField("item_ctr", builder.createDouble()));
        fields.add(builder.createField("is_new_user", builder.createInt()));
        fields.add(builder.createField("embedding_inner_product", builder.createDouble()));
        StructType schema = builder.createSchema(fields);

        Row features = builder.createRow(20143, 52, 16, "movie", "1533487890", 0L, 0.69314718d, 0.00617256d, 0, 0.5);

        JavaModelServer javaModelServer = new JavaModelServer("model/inn.model.mleap.zip", schema);
        Row result = javaModelServer.forecast(features);

        for(int i = 0 ; i < result.size(); i++){
            System.out.println(result.get(i));
        }
        DenseTensor prob = result.getTensor(16).toDense();

        System.out.println(Scala2JavaConverter.pauseCtr(prob));
    }
}
