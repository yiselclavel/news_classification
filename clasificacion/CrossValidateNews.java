package irony_detection.clasificacion;

import com.aliasi.classify.Classifier;
import com.aliasi.classify.ClassifierEvaluator;
import com.aliasi.classify.Classification;
import com.aliasi.classify.ConfusionMatrix;
import com.aliasi.classify.DynamicLMClassifier;
import com.aliasi.classify.JointClassification;
import com.aliasi.classify.LMClassifier;
import com.aliasi.classify.XValidatingClassificationCorpus;

import com.aliasi.lm.NGramProcessLM;

import com.aliasi.util.AbstractExternalizable;
import com.aliasi.util.Files;

import java.io.File;
import java.io.IOException;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class CrossValidateNews {

    private static File TRAINING_DIR
        = new File("D:/Asignaturas/Tesis/LingPipe/clasificadordenoticias/data/fourNewsGroups/4news-train");

    private static File TESTING_DIR
        =  new File("D:/Asignaturas/Tesis/LingPipe/clasificadordenoticias/data/fourNewsGroups/4news-test");

    private static String[] CATEGORIES
        = { "Internacionales",
            "Nacionales",
            "Holguin",
            "Suplementos",
            "Opinion",
            "Especiales" };

    private static int NGRAM_SIZE = 6;

    private static int NUM_FOLDS = 10;

    public static void main(String[] args)
        throws ClassNotFoundException, IOException {

        XValidatingClassificationCorpus<CharSequence> corpus
            = new XValidatingClassificationCorpus<CharSequence>(NUM_FOLDS);

        System.out.println("Reading data.");
        // read data for train and test both
        for (String category : CATEGORIES) {
            Classification c = new Classification(category);

            File trainCatDir = new File(TRAINING_DIR,category);
            for (File trainingFile : trainCatDir.listFiles()) {
                String text = Files.readFromFile(trainingFile,"ISO-8859-1");
                corpus.handle(text,c);
            }

            File testCatDir = new File(TESTING_DIR,category);
            for (File testFile : testCatDir.listFiles()) {
                String text = Files.readFromFile(testFile,"ISO-8859-1");
                corpus.handle(text,c);
            }
        }

        System.out.println("Num instances=" + corpus.numInstances() + ".");
        System.out.println("Permuting corpus.");
        corpus.permuteCorpus(new Random()); // destroys runs of categories

        System.out.printf("%5s  %10s\n","FOLD","ACCU");
        for (int fold = 0; fold < NUM_FOLDS; ++fold) {
            corpus.setFold(fold);

            DynamicLMClassifier<NGramProcessLM> classifier
                = DynamicLMClassifier.createNGramProcess(CATEGORIES,NGRAM_SIZE);
            corpus.visitTrain(classifier);
            @SuppressWarnings("unchecked") // know type is ok by compilation
            Classifier<CharSequence,JointClassification> compiledClassifier
                = (Classifier<CharSequence,JointClassification>)
                AbstractExternalizable.compile(classifier);

            ClassifierEvaluator<CharSequence,JointClassification> evaluator
                = new ClassifierEvaluator<CharSequence,JointClassification>(compiledClassifier,
                                                                            CATEGORIES);
            corpus.visitTest(evaluator);
            System.out.printf("%5d  %4.2f +/- %4.2f\n", fold,
                              evaluator.confusionMatrix().totalAccuracy(),
                              evaluator.confusionMatrix().confidence95());
        }

    }
}
