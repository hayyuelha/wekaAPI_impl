/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

import java.util.logging.Level;
import java.util.logging.Logger;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.Instances;
import weka.core.FastVector;
import weka.core.Attribute;
import weka.classifiers.Classifier;
import weka.classifiers.trees.J48;
import weka.classifiers.Evaluation;
import java.util.Random;
import weka.core.SerializationHelper;


/**
 *
 * @author HP
 */
public class homemadeWEKA {
    public static Instances loadData (String filename){
        
        Instances data = null;
        try {
            data = DataSource.read(filename);
            if (data.classIndex() == -1)
                data.setClassIndex(data.numAttributes() - 1);
        } catch (Exception ex) {
            Logger.getLogger(homemadeWEKA.class.getName()).log(Level.SEVERE, null, ex);
        }
         
        return data;
    }
    
    public static void treeLearning (Instances data) throws Exception{
//        Evaluation eval = new Evaluation(data);
        Classifier model_tree = (Classifier)new J48();
        model_tree.buildClassifier(data); // build classifier pruned tree, succeed
//        String forPredictionsPrinting = null;
//        double[] print_eval;
//        print_eval = eval.evaluateModel(model_tree, data, forPredictionsPrinting);
//        System.out.println(forPredictionsPrinting);
        save_model(model_tree);
//        return model_tree; 
    }
    
    public static void treeLearning_crossVal (Instances data) throws Exception{
        Evaluation eval = new Evaluation(data);
        J48 tree = new J48();
        eval.crossValidateModel(tree, data, 10, new Random(1));
        save_modelWithEval(tree, eval);
//        System.out.println(eval.toSummaryString("\nResult of tree learning with cross validation 10 folds\n \n",false));
    }
    
    public static void save_model (Classifier cls) throws Exception{
        SerializationHelper.write("j48.model", cls);
    }
    
    public static void save_modelWithEval (Classifier cls, Evaluation eval) throws Exception{
        Object[] o = new Object[2];
        o[0] = cls;
        o[1] = eval;
        SerializationHelper.writeAll("j48_10folds.model", o);
    }
    
    public static Classifier load_model (String mdl) throws Exception{
        Classifier cls = (Classifier) SerializationHelper.read(mdl);
        return cls; 
    }
    
    public static void reevaluateModel (Instances data_train, Instances data_test, Classifier cls) throws Exception{
        Evaluation eval = new Evaluation(data_train);
        eval.evaluateModel(cls, data_test);
        System.out.println(eval.toSummaryString("\nResults\n\n", false));
    }
    
    public static void main (String[] args) throws Exception{
        Instances file_in = loadData("weather.nominal.arff"); //succeed
        treeLearning_crossVal(file_in);
        Classifier treeModel = load_model("j48_10folds.model");
        reevaluateModel(file_in, file_in, treeModel);
//        System.out.println(tree_result.toString());
//        System.out.println("~~~~~~~~~~~~~~~~~~~~~~");
//        treeLearning_crossVal(file_in);
    }
}
