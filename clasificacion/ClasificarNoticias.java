package irony_detection.clasificacion;

import irony_detection.util.FilesUtil;
import irony_detection.util.Noticia;
import irony_detection.util.Categoria;
import irony_detection.procesamiento.ProcesarNoticias;
import com.aliasi.classify.Classifier;
import com.aliasi.classify.ClassifierEvaluator;
import com.aliasi.classify.ConfusionMatrix;
import com.aliasi.classify.DynamicLMClassifier;
import com.aliasi.classify.JointClassification;
import com.aliasi.util.AbstractExternalizable;
import java.io.File;
import java.io.IOException;
import com.aliasi.util.Files;
import java.util.LinkedList;

public class ClasificarNoticias {

    public static File originalTRAINING_DIR = new File("datos/Documentos de Entrenamiento Originales"); //noticias originales a preprocesar, extraídas de la BD
    public static File TRAINING_DIR = new File("datos/Documentos de Entrenamiento"); //noticias preprocesados para entrenar
    public static File originalExternalNews_DIR = new File("datos/Noticias Externas Originales"); //dir de noticias externas a preprocesar
    public static File ExternalNews_DIR = new File("datos/Noticias Externas Procesadas"); //dir de noticias externas a clasificar
    public static File[] ExternalNews = null; //noticias externas originales
    // public static String[] CATEGORIES = null;
    private static String[] CATEGORIES = {
        "Cultura",
        "Deporte",
        "Especiales",
        "Holguin",
        "Internacional",
        "Nacional",
        "Opinion",
        "Salud"};
    private static int NGRAM_SIZE = 6;
    private static DynamicLMClassifier classifier;
    public static StringBuilder out;

    public static void train() throws ClassNotFoundException, IOException {

        //--------le pasaria un arreglo con las noticias para entrenar
        ProcesarNoticias.preprocesarCorpusEntrenamiento(originalTRAINING_DIR.listFiles());

        classifier = DynamicLMClassifier.createNGramProcess(CATEGORIES, NGRAM_SIZE);

        int j;
        for (int i = 0; i < CATEGORIES.length; ++i) {
            File classDir = new File(TRAINING_DIR, CATEGORIES[i]);
            String[] trainingFiles = classDir.list();
            for (j = 0; j < trainingFiles.length; ++j) {
                File file = new File(classDir, trainingFiles[j]);
                String text = Files.readFromFile(file, "ISO-8859-1");
                classifier.train(CATEGORIES[i], text);
            }
        }
    }

    private static LinkedList<Noticia> classify(LinkedList<Noticia> news, File dir) throws ClassNotFoundException, IOException {
        train(); //entrenar clasificador

        ProcesarNoticias.preprocesarCorpus(news, dir); //preprocesar noticias

        //compiling
        System.out.println("Compiling");
        @SuppressWarnings("unchecked") // we created object so know it's safe
        Classifier<CharSequence, JointClassification> compiledClassifier = (Classifier<CharSequence, JointClassification>) AbstractExternalizable.compile(classifier);

        LinkedList<String> bestCategories = null;
        Noticia n = null;
        LinkedList<Noticia> clasificacion = new LinkedList<Noticia>();

        //testing        
        ClassifierEvaluator<CharSequence, JointClassification> evaluator = new ClassifierEvaluator<CharSequence, JointClassification>(compiledClassifier, CATEGORIES);

        int j, k;
        String text;
        JointClassification jc;
        for (int i = 0; i < CATEGORIES.length; ++i) {
//            File classDir = new File(TESTING_DIR,CATEGORIES[i]);
//            String[] testingFiles = classDir.list();
            for (j = 0; j < news.size(); ++j) {
//            if (news.get(j) != null) {
                n = news.get(j);
                text = Files.readFromFile(n.getNoticia(), "ISO-8859-1");
                System.out.print("Testing on " + n.getNoticia().getName() + "\n");
                evaluator.addCase(CATEGORIES[i], text);
                jc = compiledClassifier.classify(text);

                //FILTRAR por el score más de -0.9 (0.5)
                bestCategories = new LinkedList<String>();
                for (k = 0; k < jc.size(); ++k) {
                    System.out.println(jc.score(k));
                    if (jc.score(k) > -9.0d) {////////////////   2    ///////////////////////////////////
                        //System.out.println("mayor");
                        bestCategories.add(jc.category(k));
                    }
                }

                System.out.println("Got best category of: " + jc.bestCategory());
                System.out.println("---------------");

                n.setCategoriasPropuestas(bestCategories);
//                n = new Noticia(j, news[j], bestCategories);
                clasificacion.add(n);
//            }
            }
        }
        ConfusionMatrix confMatrix = evaluator.confusionMatrix();
//        System.out.println("total correctos: " + confMatrix.totalCorrect());
//        System.out.println("total clasificacciones: " + confMatrix.totalCount());

//        //imprimir matriz
//        System.out.println("confusion matriz:");
//        int[][] m = confMatrix.matrix();
//        for (int i = 0; i < m.length; i++) {
//            for (j = 0; j < m[0].length; j++) {
//                System.out.print(m[i][j] + " ");
//            }
//            System.out.println();
//        }
      //  System.out.println("Total Accuracy: " + confMatrix.totalAccuracy());
        return clasificacion;
    }

    public static LinkedList<Noticia> classifyNews() throws Exception {
        ExternalNews = originalExternalNews_DIR.listFiles();
        LinkedList<Noticia> n = new LinkedList<Noticia>();
        String text;
        for (int i = 0; i < ExternalNews.length; i++) {
            if (ExternalNews[i] != null) {
                text = Files.readFromFile(ExternalNews[i], "ISO-8859-1");
                n.add(new Noticia(i, ExternalNews[i], ExternalNews[i].getName(), text, ExternalNews[i].lastModified()));
            }
        }
        return classify(n, ExternalNews_DIR);
    }

}
