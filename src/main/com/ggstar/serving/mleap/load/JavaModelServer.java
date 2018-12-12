package com.ggstar.serving.mleap.load;
import ml.combust.mleap.core.types.*;
import ml.combust.mleap.runtime.MleapContext;
import ml.combust.mleap.runtime.javadsl.BundleBuilder;
import ml.combust.mleap.runtime.javadsl.ContextBuilder;
import ml.combust.mleap.runtime.javadsl.LeapFrameBuilder;
import ml.combust.mleap.runtime.frame.*;

import java.io.File;
import java.util.ArrayList;

public class JavaModelServer {

    private String modelPath;
    private StructType dataSchema;

    private Transformer model;

    public JavaModelServer(String modelPath, StructType dataSchema){
        this.modelPath = modelPath;
        this.dataSchema = dataSchema;
    }

    public void loadModel(){
        MleapContext mleapContext = new ContextBuilder().createMleapContext();
        BundleBuilder bundleBuilder = new BundleBuilder();
        this.model = bundleBuilder.load(new File(this.modelPath), mleapContext).root();
    }

    public Row forecast(Row features){
        if (this.model == null){
            loadModel();
        }
        if (features == null){
            System.err.println("features is null");
            return null;
        }

        LeapFrameBuilder builder = new LeapFrameBuilder();

        ArrayList<Row> rows = new ArrayList<>();
        rows.add(features);
        DefaultLeapFrame frame = builder.createFrame(dataSchema, rows);
        DefaultLeapFrame result = this.model.transform(frame).get();
        return result.dataset().head();
    }
}
