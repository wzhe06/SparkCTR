package com.ggstar.example;

import java.io.Serializable;

public class Sample implements Serializable {
    private int user_id;
    private int item_id;
    private int category_id;
    private String content_type;
    private String timestamp;
    private long user_item_click;
    private double user_item_imp;
    private double item_ctr;
    private int is_new_user;
    private double embedding_inner_product;

    public Sample(){

    }

    public Sample(int user_id,
                  int item_id,
                  int category_id,
                  String content_type,
                  String timestamp,
                  long user_item_click,
                  double user_item_imp,
                  double item_ctr,
                  int is_new_user,
                  double embeddingInnerProduct){
        this.user_id = user_id;
        this.item_id = item_id;
        this.category_id = category_id;
        this.content_type = content_type;
        this.timestamp = timestamp;
        this.user_item_click = user_item_click;
        this.user_item_imp = user_item_imp;
        this.item_ctr = item_ctr;
        this.is_new_user = is_new_user;
        this.embedding_inner_product = embedding_inner_product;
    }

    public int getUser_id() {
        return user_id;
    }

    public void setUser_id(int user_id) {
        this.user_id = user_id;
    }

    public int getItem_id() {
        return item_id;
    }

    public void setItem_id(int item_id) {
        this.item_id = item_id;
    }

    public int getCategory_id() {
        return category_id;
    }

    public void setCategory_id(int category_id) {
        this.category_id = category_id;
    }

    public String getContent_type() {
        return content_type;
    }

    public void setContent_type(String content_type) {
        this.content_type = content_type;
    }

    public String getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(String timestamp) {
        this.timestamp = timestamp;
    }

    public long getUser_item_click() {
        return user_item_click;
    }

    public void setUser_item_click(long user_item_click) {
        this.user_item_click = user_item_click;
    }

    public double getUser_item_imp() {
        return user_item_imp;
    }

    public void setUser_item_imp(double user_item_imp) {
        this.user_item_imp = user_item_imp;
    }

    public double getItem_ctr() {
        return item_ctr;
    }

    public void setItem_ctr(double item_ctr) {
        this.item_ctr = item_ctr;
    }

    public int getIs_new_user() {
        return is_new_user;
    }

    public void setIs_new_user(int is_new_user) {
        this.is_new_user = is_new_user;
    }

    public double getEmbedding_inner_product() {
        return embedding_inner_product;
    }

    public void setEmbedding_inner_product(double embedding_inner_product) {
        this.embedding_inner_product = embedding_inner_product;
    }
}
