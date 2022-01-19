package clasificacion_noticias.procesamiento;

import clasificacion_noticias.clasificacion.ClasificarNoticias;
import clasificacion_noticias.util.FilesUtil;

import clasificacion_noticias.util.Noticia;
import java.util.*;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintStream;
import java.io.Reader;

//Lucene para preprocesar los documentos
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.StandardAnalyzer;
import org.apache.lucene.analysis.Token;
import org.apache.lucene.analysis.TokenStream;

//LingPipe para clasificar
import com.aliasi.classify.BernoulliClassifier;
import com.aliasi.classify.BinaryLMClassifier;
import com.aliasi.classify.Classification;
import com.aliasi.classify.Classifier;
import com.aliasi.classify.ClassifierEvaluator;
import com.aliasi.classify.DynamicLMClassifier;
import com.aliasi.classify.JointClassification;
import com.aliasi.classify.KnnClassifier;
import com.aliasi.classify.LogisticRegressionClassifier;
import com.aliasi.classify.NaiveBayesClassifier;
import com.aliasi.classify.XValidatingClassificationCorpus;
import com.aliasi.stats.RegressionPrior;
import com.aliasi.stats.AnnealingSchedule;
import com.aliasi.classify.PerceptronClassifier;
import com.aliasi.tokenizer.RegExTokenizerFactory;
import com.aliasi.classify.ScoredClassification;
import com.aliasi.classify.TfIdfClassifierTrainer;
import com.aliasi.classify.TradNaiveBayesClassifier;
import com.aliasi.corpus.ClassificationHandler;
import com.aliasi.corpus.Corpus;
import com.aliasi.matrix.PolynomialKernel;
import com.aliasi.tokenizer.IndoEuropeanTokenizerFactory;
import com.aliasi.tokenizer.TokenFeatureExtractor;
import com.aliasi.tokenizer.Tokenizer;
import com.aliasi.tokenizer.TokenizerFactory;
import com.aliasi.util.AbstractExternalizable;
import com.aliasi.util.FeatureExtractor;
import com.aliasi.util.Files;
import com.aliasi.util.ObjectToDoubleMap;
import com.aliasi.util.Strings;

////java
import java.io.Serializable;

public class ProcesarNoticias {

    private static final String[] STOP_WORDS = {"a", "ante", "antes", "bajo", "cabe", "con", "contra", "de", "desde", "en", "entre", "hacia", "hasta", "para", "por", "según",
        "sin", "sobre", "tras", "y", "e", "ni", "o", "u", "pero", "empero", "mas", "sino", "porque", "conque", "puesto", "que", "pues",
        "si", "aunque", "tan", "como", "conforme", "cuando", "mientras", "luego", "así", "quien", "cual", "cuyo", "donde", "cuanto",
        "aquí", "allí", "ahí", "acá", "allá", "cerca", "lejos", "arriba", "abajo", "encima", "debajo", "detrás", "enfrente", "fuera",
        "dentro", "ayer", "anteayer", "entonces", "ya", "hoy", "ahora", "después", "mañana", "aún", "todavía", "siempre", "nunca",
        "jamás", "tarde", "temprano", "pronto", "bien", "mal", "apenas", "adrede", "despacio", "recio", "duro", "fuerte", "alto",
        "más", "menos", "mucho", "poco", "casi", "bastante", "harto", "demasiado", "tanto", "muy", "sólo", "solo", "solamente", "sumamente",
        "tremendamente", "primeramente", "posteriormente", "cierto", "cierta", "ciertas", "ciertamente", "verdad", "verdadero", "verdadera",
        "verdaderamente", "indudablemente", "también", "seguramente", "sí", "no", "tampoco", "quizás", "acaso", "yo", "tú", "usted", "él", "ella",
        "ello", "nosotros", "nosotras", "vosotros", "vosotras", "ustedes", "ellos", "ellas", "mío", "mía", "míos", "mías", "tuyo", "tuya",
        "tuyos", "tuyas", "suyo", "suya", "suyos", "suyas", "nuestro", "nuestra", "nuestros", "nuestras", "mi", "su", "sus", "tu",
        "este", "éste", "esté", "esta", "ésta", "está", "estamos", "estáis", "están", "esto", "estos", "éstos", "estas", "éstas",
        "eso", "ese", "ése", "esa", "ésa", "esos", "ésos", "esas", "ésas", "aquel", "aquella", "aquellos", "aquellas", "quienes", 
        "cuya", "cuyos", "cuyas", "quién", "quiénes", "qué", "cuál", "cuáles", "cuándo", "cuánto", "cómo", "alguien", "nadie", "algo", "nada",
        "alguno", "algunos", "alguna", "algunas", "cualquiera", "quienquiera", "todo", "ninguno", "ninguna", "varios", "ciertos", "otro", "mí", "conmigo",
        "ti", "contigo", "consigo", "me", "te", "lo", "la", "le", "se", "nos", "os", "los", "las", "les", "el", "al", "un", "uno",
        "una", "unos", "unas", "algún", "ser", "es", "soy", "eres", "somos", "sois", "estoy", "atrás", "por qué", "estado", "estaba",
        "siendo", "ambos", "poder", "puede", "puedo", "podemos", "podéis", "pueden", "fui", "fue", "fuimos", "fueron", "hacer", "hago", "hace",
        "hacemos", "hacéis", "hacen", "cada", "fin", "incluso", "primero", "conseguir", "consigue", "consigues", "conseguimos", "consiguen",
        "ir", "voy", "va", "vamos", "vais", "van", "vaya", "bueno", "ha", "tener", "tengo", "tiene", "tenemos", "tenéis", "tienen", "saber",
        "sabes", "sabe", "sabemos", "sabéis", "saben", "último", "largo", "haces", "muchos", "intentar", "intento", "intenta", "intentas",
        "intentamos", "intentáis", "intentan", "dos", "usar", "uso", "usas", "usa", "usamos", "usáis", "usan", "emplear", "empleo", "empleas",
        "emplean", "empleamos", "empleáis", "valor", "era", "eras", "éramos", "eran", "modo", "podría", "podrías", "podríamos", "podrían",
        "podríais", "del"};

    //Crea documentos nuevos sin las StopWords.
    //Le aplica el Stemming en español
    //Utiliza como Analyzer: StandarAnalyzer
    public static void preprocesarCorpus(LinkedList<Noticia> news, File dir) {
        int numeroFichero = 0;
        //El analizador sera StandardAnalyzer filtra StandarTokenizer.
        //Este a diferencia de SimpleAnalyzer no me divide los correos y hostname en token diferentes. (eg. mrodriguezh@facinf.uho.edu.cu)
        //Ademas de los puntos que no le sigan espacio en blanco lo toma todo como un token (eg. ee.uu)
        //La convierte a minuscula y elimina los stopsWords
        Analyzer anlyzer = new StandardAnalyzer(STOP_WORDS);

        //Recorro la lista que contiene los ficheros .txt
        for (Noticia n : news) {
            File f = n.getNoticia();
            FileOutputStream fileOutputStream = null;
            try {
                numeroFichero++;
                fileOutputStream = new FileOutputStream(dir + "/" + f.getName());
                PrintStream p = new PrintStream(fileOutputStream);
                //Verifico que lo que voy a analizar solo son ficheros
                if (f.isFile()) {
                    //Imprimo los nombres de los ficheros .txt
//              System.out.println(temp.getName());
                    //Leer el fichero
                    Reader reader = new FileReader(f);
                    //Flujo de texto del fichero .txt
                    TokenStream stream = anlyzer.tokenStream(f.getName(), reader);
                    //Lleva las palabras a la raiz y quita las tildes
                    SpanishStemFilter streamRaiz = new SpanishStemFilter(stream);
                    //Obtener el proximo token
                    Token token = streamRaiz.next(new Token());
                    while (token != (null)) {
                        p.println(token.termText());
                        token = streamRaiz.next(new Token());
                    }
                }
            } catch (Exception ex) {
                System.out.println(ex.getMessage());
            } finally {
                try {
                    fileOutputStream.close();
                } catch (IOException ex) {
                    System.out.println(ex.getMessage());
                }
            }
        }
    }

    public static void preprocesarCorpusEntrenamiento(File[] file) {
        int numeroFichero = 0;
        //El analizador sera StandardAnalyzer filtra StandarTokenizer.
        //Este a diferencia de SimpleAnalyzer no me divide los correos y hostname en token diferentes. (eg. mrodriguezh@facinf.uho.edu.cu)
        //Ademas de los puntos que no le sigan espacio en blanco lo toma todo como un token (eg. ee.uu)
        //La convierte a minuscula y elimina los stopsWords
        Analyzer anlyzer = new StandardAnalyzer(STOP_WORDS);

        for (int i = 0; i < file.length; ++i) {
            File[] f = file[i].listFiles();
            for (int j = 0; j < f.length; ++j) {
                try {
                    numeroFichero++;

                    FileOutputStream fileOutputStream = new FileOutputStream(ClasificarNoticias.TRAINING_DIR + "/" + file[i].getName() + "/" + f[j].getName());

                    PrintStream p = new PrintStream(fileOutputStream);

                    //Obtengo el primer fichero .txt
                    File fileF = f[j];

                    //File temp = new File(fileF.getAbsolutePath() +  "\\" + string);
                    //Verifico que lo que voy a analizar solo son ficheros
                    if (fileF.isFile()) {
                        //Imprimo los nombres de los ficheros .txt
//             System.out.println(fileF.getName());
                        //Leer el fichero
                        Reader reader = new FileReader(fileF);
                        //Flujo de texto del fichero .txt
                        TokenStream stream = anlyzer.tokenStream(fileF.getName(), reader);
                        //Lleva las palabras a la raiz y quita las tildes
                        SpanishStemFilter streamRaiz = new SpanishStemFilter(stream);
                        //Obtener el proximo token
                        Token token = streamRaiz.next(new Token());
                        while (token != (null)) {
                            p.println(token.termText());
                            token = streamRaiz.next(new Token());
                        }

                    }
                } catch (Exception ex) {
                    System.out.println(ex.getMessage());
                }
            }
        }

//        }
    }

    //DynamicLMClassifier with LingPipe
    public void clasificacionDynamicLMClassifier(File TRAINING_DIR, File TESTING_DIR, String[] CATEGORIES, int NGRAM_SIZE) throws ClassNotFoundException, IOException {
        DynamicLMClassifier classifier = DynamicLMClassifier.createNGramProcess(CATEGORIES, NGRAM_SIZE);

        //training
        for (int i = 0; i < CATEGORIES.length; ++i) {
            File classDir = new File(TRAINING_DIR, CATEGORIES[i]);
            if (!classDir.isDirectory()) {
                String msg = "Could not find training directory=" + classDir + "\nHave you unpacked 4 newsgroups?";
                System.out.println(msg); // in case exception gets lost in shell
                throw new IllegalArgumentException(msg);
            }

            String[] trainingFiles = classDir.list();
            for (int j = 0; j < trainingFiles.length; ++j) {
                File fileF = new File(classDir, trainingFiles[j]);
                String text = Files.readFromFile(fileF, "ISO-8859-1");
                System.out.println("Training on " + CATEGORIES[i] + "/" + trainingFiles[j]);
                classifier.train(CATEGORIES[i], text);
            }
        }

        //compiling
        System.out.println("Compiling");
        @SuppressWarnings("unchecked") // we created object so know it's safe
        Classifier<CharSequence, JointClassification> compiledClassifier = (Classifier<CharSequence, JointClassification>) AbstractExternalizable.compile(classifier);

        //testing
        File classDir = new File(TESTING_DIR.getAbsolutePath());//,CATEGORIES[i]
        String[] testingFiles = classDir.list();
        for (int j = 0; j < testingFiles.length; ++j) {
            String text = Files.readFromFile(new File(classDir, testingFiles[j]), "ISO-8859-1");
            System.out.print("Testing: " + "/" + testingFiles[j] + " ");
            JointClassification jc = compiledClassifier.classify(text);
            String bestCategory = jc.bestCategory();
            String details = jc.toString();
            System.out.println("Got best category of: " + bestCategory);
            System.out.println(details);
            System.out.println("---------------");
        }
    }

    //DynamicLMClassifier with LingPipe
    public void clasificacionDynamicLMClassifierCross_Validation(File TRAINING_DIR, File TESTING_DIR, String[] CATEGORIES, int NGRAM_SIZE, int NUM_FOLDS) throws ClassNotFoundException, IOException {
        XValidatingClassificationCorpus<CharSequence> corpus = new XValidatingClassificationCorpus<CharSequence>(NUM_FOLDS);
        System.out.println("Reading data.");
        // read data for train and test both

        //Training
        for (int i = 0; i < CATEGORIES.length; ++i) {
            Classification c = new Classification(CATEGORIES[i]);
            File classDir = new File(TRAINING_DIR, CATEGORIES[i]);
            if (!classDir.isDirectory()) {
                String msg = "Could not find training directory=" + classDir + "\nHave you unpacked 4 newsgroups?";
                System.out.println(msg); // in case exception gets lost in shell
                throw new IllegalArgumentException(msg);
            }

            String[] trainingFiles = classDir.list();
            for (int j = 0; j < trainingFiles.length; ++j) {
                File fileF = new File(classDir, trainingFiles[j]);
                String text = Files.readFromFile(fileF, "ISO-8859-1");
                // System.out.println("Training on " + CATEGORIES[i] + "/" + trainingFiles[j]);
                corpus.handle(text, c);
            }
        }

        System.out.println("Num instances=" + corpus.numInstances() + ".");
        System.out.println("Permuting corpus.");
        corpus.permuteCorpus(new Random()); // destroys runs of categories

        for (int fold = 0; fold < NUM_FOLDS; ++fold) {
            corpus.setFold(fold);

            DynamicLMClassifier classifier = DynamicLMClassifier.createNGramProcess(CATEGORIES, NGRAM_SIZE);
            corpus.visitTrain(classifier);
            @SuppressWarnings("unchecked") // know type is ok by compilation
            Classifier<CharSequence, JointClassification> compiledClassifier = (Classifier<CharSequence, JointClassification>) AbstractExternalizable.compile(classifier);

            ClassifierEvaluator<CharSequence, JointClassification> evaluator = new ClassifierEvaluator<CharSequence, JointClassification>(compiledClassifier, CATEGORIES);
            corpus.visitTest(evaluator);
            System.out.println("toString:" + " " + evaluator.confusionMatrix().toString());
        }
    }

    //TfIdfClassifierTrainer with LingPipe
    public void clasificacionTF_IDF(File TRAINING_DIR, File TESTING_DIR, String[] CATEGORIES, LinkedList FuenteExterna) throws ClassNotFoundException, IOException {
        /********************** Construction *********************************************************************************************
         * TfIdfClassifierTrainer is constructed from a feature extractor of a specified type.
         * If the instance is to be compiled, the feature extractor must be either serializable or compilable.
         * producing an instance that may be trained through
         **********************************************************************************************************************************/
        /* TokenizerFactory SPACE_TOKENIZER_FACTORY = new NGramTokenizerFactory(2, 6);
        FeatureExtractor<CharSequence> trainer = new TokenFeatureExtractor(SPACE_TOKENIZER_FACTORY);
        TfIdfClassifierTrainer<CharSequence> tf_idfClassifier = new TfIdfClassifierTrainer<CharSequence>(trainer); */
        TokenFeatureExtractor featureExtractor = new TokenFeatureExtractor(new IndoEuropeanTokenizerFactory());
        TfIdfClassifierTrainer<CharSequence> trainer = new TfIdfClassifierTrainer<CharSequence>(featureExtractor);


        /********************** Training **************************************************************************************************
         * Categories may be added dynamically.  The initial classifier will be empty and not defined for any categories.
         *
         * A TF/IDF classifier trainer is trained through the ClassificationHandler. Specifically, the method handle(E,Classification) is called, the generic
         * object being the training instance and the classification being a simple first-best classification.
         *
         * For multiple training examples of the same category, their feature vectors are added together to produce the raw category vectors.
         *************************************************************************************************************************************/
        for (int i = 0; i < CATEGORIES.length; ++i) {
            File classDir = new File(TRAINING_DIR, CATEGORIES[i]);
            if (!classDir.isDirectory()) {
                String msg = "Could not find training directory=" + classDir + "\nHave you unpacked 4 newsgroups?";
                System.out.println(msg);
                throw new IllegalArgumentException(msg);
            }

            String[] trainingFiles = classDir.list();
            for (int j = 0; j < trainingFiles.length; ++j) {
                File fileF = new File(classDir, trainingFiles[j]);
                String text = Files.readFromFile(fileF, "ISO-8859-1");
                System.out.println("Training on " + CATEGORIES[i] + "/" + trainingFiles[j]);

                Classification classification = new Classification(CATEGORIES[i]);
                //El metodo handle entrena el clasificador en el objeto especificado(text) con la clasificacion(classification) especificada
                trainer.handle(text, classification);
            }
        }
        /***************** Compilation **************************************************************************************************************************
         * At any point, a TF/IDF classifier may be compiled to an object output stream.  The object read back in will be an instance of Classifier<E,ScoredClassification>.
         * During compilation, the feature extractor will be compiled if it's compilable, or serialized if it's serializable but not compilable.
         * If the feature extractor is neither compilable nor serializable, compilation will throw an error.
         *************************************************************************************************************************************************************/
        System.out.println("Compiling");
        @SuppressWarnings("unchecked") // we created object so know it's safe
        Classifier<CharSequence, ScoredClassification> compiledClassifier = (Classifier<CharSequence, ScoredClassification>) AbstractExternalizable.compile(trainer);
        /********************** Classification ************************************************************************************************
         * The compiled models perform scored classification.  That is, they implement the method classify(E) to return a ScoredClassification.
         * The scores assigned to the different categories are normalized dot products after term frequency and inverse document frequency weighting.
         * Suppose training supplied n training categories cat[0], ..., cat[n-1], with associated raw feature vectors v[0], ..., v[n-1].
         * The dimensions of these vectors are the features, so that if f is a feature, v[i][f] is the raw score for the feature f in category cat[i].
         * First, the inverse document frequency weighting of each term is defined:
         *     idf(f) = ln (df(f)/n)
         * where df(f) is the document frequency of feature f, defined to be the number of distinct categories in which feature f is defined.
         * This has the effect of upweighting the scores of features that occur in few categories and downweighting the scores of features that occur in many categories
         * Term frequency normalization dampens the term frequencies using square roots:
         *     tf(x) = sqrt(x)
         * This produces a linear relation in pairwise growth rather than the usual quadratic one derived from a simple cross-product.
         *
         *The weighted feature vectors are as follows:
         *     v'[i][f] = tf(v[i][f]) * idf(f)
         * Given an instance to classify, first the feature extractor is used to produce a raw feature vector x. This is then normalized in the same
         * way as the document vectors v[i], namely:
         *     x'[f] = tf(x[f]) * idf(f)
         * The resulting query vector x'is then compared against each normalized document vector v'[i] using vector cosine, which defines its classification score:
         *      score(v'[i],x') = cos(v'[i],x')  = v'[i] * x'/(length(v'[i])*length(x'))
         * where v'[i]*x' is the vector dot product:
         *     Σf v'[i][f]*x'[f]
         * and where the length of a vector is defined to be the square root of its dot product with itself:
         *     length(y) = sqrt(y*y)
         * Cosine scores will vary between -1 and 1.
         * The cosine is only 1 between two vectors if they point in the same direction; that is, one is a positive scalar product of the other.
         * The cosine is only -1 between two vectors if they point in opposite direction; that is, one is a negative scalar product of the other.
         * The cosine is 0 for two vectors that are orthogonal, that is, at right angles to each other.
         * If all the values in all of the category vectors and the query vector are positive, cosine will run between 0 and 1.
         **********************************************************************************************************************************************************************/
        //testing
        String bestCategory = "";
        File classDir = new File(TESTING_DIR.getAbsolutePath());
        String[] testingFiles = classDir.list();
        for (int j = 0; j < testingFiles.length; ++j) {
            String text = Files.readFromFile(new File(classDir, testingFiles[j]), "ISO-8859-1");
            System.out.print("Testing :" + "/" + testingFiles[j] + " ");

            //Este metodo classify es implementado y es el que tiene en cuenta el TF_IDF normalizado
            ScoredClassification jc = compiledClassifier.classify(text);
            bestCategory = jc.bestCategory();
            System.out.println("Got best category of: " + bestCategory);
            String details = jc.toString();
            System.out.println(details);
            System.out.println("---------------");
        }
        /******************* Warning: ***************************************************************************************************************************
         * Because of floating-point arithmetic rounding, these results about signs and bounds are not strictly guaranteed to hold;
         * instances may return cosines slightly below -1 or above 1, or not return exactly 0 for orthogonal vectors.
         ***********************************************************************************************************************************************************/
        /****************** Serialization ***********************************************************************************************************************
         * A TF/IDF classifier trainer may be serialized at any point.
         * The object read back in will be an instance of the same class with the same parametric type for the objects being classified.
         * During serialization, the feature extractor will be serialized if it's serializable, or compiled if it's compilable but not serializable.
         * If the feature extractor is neither serializable nor compilable, serialization will throw an error.
         **********************************************************************************************************************************************************/
        /****************** Reverse Indexing *******************************************************************************************************************************
         * The TF/IDF classifier indexes instances by means of their feature values.
         ********************************************************************************************************************************************************************/
    }
//TfIdfClassifierTrainer with LingPipe

    public void clasificacionTF_IDFCross_Validation(File TRAINING_DIR, File TESTING_DIR, String[] CATEGORIES, LinkedList FuenteExterna, int NUM_FOLDS) throws ClassNotFoundException, IOException {
        XValidatingClassificationCorpus<CharSequence> corpus = new XValidatingClassificationCorpus<CharSequence>(NUM_FOLDS);

        for (int i = 0; i < CATEGORIES.length; ++i) {
            Classification c = new Classification(CATEGORIES[i]);
            File classDir = new File(TRAINING_DIR, CATEGORIES[i]);
            if (!classDir.isDirectory()) {
                String msg = "Could not find training directory=" + classDir + "\nHave you unpacked 4 newsgroups?";
                System.out.println(msg);
                throw new IllegalArgumentException(msg);
            }

            String[] trainingFiles = classDir.list();
            for (int j = 0; j < trainingFiles.length; ++j) {
                File fileF = new File(classDir, trainingFiles[j]);
                String text = Files.readFromFile(fileF, "ISO-8859-1");
                corpus.handle(text, c);
            }
        }

        System.out.println("Num instances=" + corpus.numInstances() + ".");
        System.out.println("Permuting corpus.");
        corpus.permuteCorpus(new Random()); // destroys runs of categories

        for (int fold = 0; fold < NUM_FOLDS; ++fold) {
            corpus.setFold(fold);

            TfIdfClassifierTrainer<CharSequence> classifier = new TfIdfClassifierTrainer<CharSequence>(new TokenFeatureExtractor(new IndoEuropeanTokenizerFactory()));

            corpus.visitTrain(classifier);
            @SuppressWarnings("unchecked") // know type is ok by compilation

            Classifier<CharSequence, JointClassification> compiledClassifier = (Classifier<CharSequence, JointClassification>) AbstractExternalizable.compile(classifier);
            ClassifierEvaluator<CharSequence, JointClassification> evaluator = new ClassifierEvaluator<CharSequence, JointClassification>(compiledClassifier, CATEGORIES);
            corpus.visitTest(evaluator);
            System.out.println("toString:" + " " + evaluator.confusionMatrix().toString());
        }
    }

    //Clasificacion con Perceptron
    public void clasificacionBinaryClassPerceptron(File TRAINING_DIR, File TESTING_DIR, String[] CATEGORIES) throws IOException {
        //Training
        PerceptronClassifier<CharSequence> svm = new PerceptronClassifier<CharSequence>(new TestFeatureExtractor(), new PolynomialKernel(3), new TestCorpus(TRAINING_DIR, CATEGORIES), BinaryLMClassifier.DEFAULT_ACCEPT_CATEGORY, 2, BinaryLMClassifier.DEFAULT_ACCEPT_CATEGORY, BinaryLMClassifier.DEFAULT_REJECT_CATEGORY);

        //El construido es Serializable porque FeatureExtractor[TokenFeatureExtractor] y KernelFunction[PolynomialKernel] lo son
        PerceptronClassifier<CharSequence> clasificar = (PerceptronClassifier<CharSequence>) AbstractExternalizable.serializeDeserialize(svm);

        //Testing
        String bestCategory = "";
        File classDir = new File(TESTING_DIR.getAbsolutePath());
        String[] testingFiles = classDir.list();
        for (int j = 0; j < testingFiles.length; ++j) {
            String text = Files.readFromFile(new File(classDir, testingFiles[j]), "ISO-8859-1");
            System.out.print("Testing :" + "/" + testingFiles[j] + " ");

            ScoredClassification jc = clasificar.classify(text);

            bestCategory = jc.bestCategory();
            System.out.println("Got best category of: " + bestCategory);
            String details = jc.toString();
            System.out.println(details);
            System.out.println("---------------");
        }
    }

    //Clasificacion LogisticRegressionClassifier (Multi Class) con LingPipe
    public void clasificacionMultiClassRegresion(String[] CATEGORIES, File TRAINING_DIR, File TESTING_DIR) throws IOException {
        Random random = new Random();

        XValidatingClassificationCorpus<CharSequence> corpus = new XValidatingClassificationCorpus<CharSequence>(10);

        //Training
        for (int i = 0; i < CATEGORIES.length; ++i) {
            File classDir = new File(TRAINING_DIR, CATEGORIES[i]);
            if (!classDir.isDirectory()) {
                String msg = "Could not find training directory=" + classDir + "\nHave you unpacked 4 newsgroups?";
                System.out.println(msg); // in case exception gets lost in shell
                throw new IllegalArgumentException(msg);
            }

            String[] trainingFiles = classDir.list();
            for (int j = 0; j < trainingFiles.length; ++j) {
                File fileF = new File(classDir, trainingFiles[j]);
                String text = Files.readFromFile(fileF, "ISO-8859-1");
                System.out.println("Training on " + CATEGORIES[i] + "/" + trainingFiles[j]);
                Classification c = new Classification(CATEGORIES[i]);
                corpus.handle(text, c);
            }
        }

        corpus.permuteCorpus(random);
        FeatureExtractor<CharSequence> featureExtractor = new TokenFeatureExtractor(new RegExTokenizerFactory("\\S+"));

        boolean addIntercept = true;
        RegressionPrior prior = RegressionPrior.noninformative();
        double initLearningRate = 0.01;
        double annealingRate = 500;
        double minImprovement = 0.001;
        int minEpochs = 2;
        int maxEpochs = 10000;
        int minFeatureCount = 2;

        LogisticRegressionClassifier<CharSequence> classifier = LogisticRegressionClassifier.train(
                featureExtractor,
                corpus,
                minFeatureCount,
                addIntercept,
                prior,
                AnnealingSchedule.inverse(initLearningRate, annealingRate),
                minImprovement,
                minEpochs, maxEpochs,
                null); // no writer feedback for test
        //Testing
        File classDir = new File(TESTING_DIR.getAbsolutePath());
        String[] testingFiles = classDir.list();
        for (int j = 0; j < testingFiles.length; ++j) {
            String text = Files.readFromFile(new File(classDir, testingFiles[j]), "ISO-8859-1");
            System.out.print("Testing :" + "/" + testingFiles[j] + " ");
            ScoredClassification jc = classifier.classify(text);
            System.out.println("bestCategory:" + " " + jc.bestCategory());
            System.out.println(jc.toString());
            System.out.println("---------------");
        }
    }
    //Clasificacion LogisticRegressionClassifier (Multi Class) con LingPipe

    public void clasificacionMultiClassRegresionCross_Validation(String[] CATEGORIES, File TRAINING_DIR, File TESTING_DIR, int NUM_FOLDS) throws IOException, ClassNotFoundException {

        XValidatingClassificationCorpus<CharSequence> corpus = new XValidatingClassificationCorpus<CharSequence>(NUM_FOLDS);

        System.out.println("Reading data.");
        // read data for train and test both
        for (int i = 0; i < CATEGORIES.length; ++i) {
            Classification c = new Classification(CATEGORIES[i]);

            File classDir = new File(TRAINING_DIR, CATEGORIES[i]);
            if (!classDir.isDirectory()) {
                String msg = "Could not find training directory=" + classDir + "\nHave you unpacked 4 newsgroups?";
                System.out.println(msg); // in case exception gets lost in shell
                throw new IllegalArgumentException(msg);
            }
            String[] trainingFiles = classDir.list();
            for (int j = 0; j < trainingFiles.length; ++j) {
                File fileF = new File(classDir, trainingFiles[j]);
                String text = Files.readFromFile(fileF, "ISO-8859-1");
                //System.out.println("Training on " + CATEGORIES[i] + "/" + trainingFiles[j]);
                corpus.handle(text, c);
            }
        }

        System.out.println("Num instances=" + corpus.numInstances() + ".");
        System.out.println("Permuting corpus.");
        corpus.permuteCorpus(new Random()); // destroys runs of categories

        //  System.out.printf("%5s  %10s\n","FOLD","ACCU");

        //corpus - Corpus of training data.
        //featureExtractor - Converter from objects to feature maps.
        FeatureExtractor<CharSequence> featureExtractor = new TokenFeatureExtractor(new RegExTokenizerFactory("\\S+"));
        //addInterceptFeature - A flag set to true if an intercept feature should be added to each input vector.
        boolean addIntercept = true;
        //prior - The prior for regularization of the regression.
        RegressionPrior prior = RegressionPrior.noninformative();
        double initLearningRate = 0.01;
        //annealingSchedule - Class to compute learning rate for each epoch.
        double annealingRate = 500;
        //minImprovement - Minimum relative improvement in error during an epoch to stop search.
        double minImprovement = 0.001;
        //minEpochs - Minimum number of search epochs.
        int minEpochs = 2;
        //maxEpochs - Maximum number of epochs.
        int maxEpochs = 10000;
        //minFeatureCount - Minimum count for features in corpus to keep feature as part of model.
        int minFeatureCount = 2;

        for (int fold = 0; fold < NUM_FOLDS; ++fold) {
            //Resetea el fold para ser el valor especificado. Iterando desde 0 hasta num_folds()-1
            corpus.setFold(fold);

            //Entreno el clasificador con el corpus indicado
            LogisticRegressionClassifier<CharSequence> classifier = LogisticRegressionClassifier.train(
                    featureExtractor,
                    corpus,
                    minFeatureCount,
                    addIntercept,
                    prior,
                    AnnealingSchedule.inverse(initLearningRate, annealingRate),
                    minImprovement,
                    minEpochs, maxEpochs,
                    null); // no writer feedback for test

            Classifier<CharSequence, JointClassification> compiledClassifier = (Classifier<CharSequence, JointClassification>) AbstractExternalizable.compile(classifier);
            ClassifierEvaluator<CharSequence, JointClassification> evaluator = new ClassifierEvaluator<CharSequence, JointClassification>(compiledClassifier, CATEGORIES);

            corpus.visitTest(evaluator);
            //System.out.printf("%5d  %4.2f +/- %4.2f\n", fold, evaluator.confusionMatrix().totalAccuracy(), evaluator.confusionMatrix().confidence95());
            System.out.println("toString:" + " " + evaluator.confusionMatrix().toString());
        }
    }

    //Naive Bayes
    public void NaiveBayesClassifier(File TRAINING_DIR, File TESTING_DIR, String[] CATEGORIES) throws IOException, ClassNotFoundException {

        NaiveBayesClassifier classifier = new NaiveBayesClassifier(CATEGORIES, new IndoEuropeanTokenizerFactory());

        //Training
        for (int i = 0; i < CATEGORIES.length; ++i) {
            File classDir = new File(TRAINING_DIR, CATEGORIES[i]);
            if (!classDir.isDirectory()) {
                String msg = "Could not find training directory=" + classDir + "\nHave you unpacked 4 newsgroups?";
                System.out.println(msg);
                throw new IllegalArgumentException(msg);
            }

            String[] trainingFiles = classDir.list();
            for (int j = 0; j < trainingFiles.length; ++j) {
                File fileF = new File(classDir, trainingFiles[j]);
                String text = Files.readFromFile(fileF, "ISO-8859-1");
                System.out.println("Training on " + CATEGORIES[i] + "/" + trainingFiles[j]);

                //El metodo handle entrena el clasificador en el objeto especificado(text) con la clasificacion(classification) especificada
                classifier.train(CATEGORIES[i], text);
            }
        }

        //Compiling
        System.out.println("Compiling");
        @SuppressWarnings("unchecked") // we created object so know it's safe
        Classifier<CharSequence, JointClassification> compiledClassifier = (Classifier<CharSequence, JointClassification>) AbstractExternalizable.compile(classifier);

        //testing
        File classDir = new File(TESTING_DIR.getAbsolutePath());
        String[] testingFiles = classDir.list();
        for (int j = 0; j < testingFiles.length; ++j) {
            String text = Files.readFromFile(new File(classDir, testingFiles[j]), "ISO-8859-1");
            System.out.print("Testing :" + "/" + testingFiles[j] + " ");
            JointClassification jc = compiledClassifier.classify(text);
            String bestCategory = jc.bestCategory();
            String details = jc.toString();
            System.out.println("Got best category of: " + bestCategory);
            System.out.println(details);
            System.out.println("---------------");
        }
    }

    //Naive Bayes Cross Validation
    //Naive Bayes
    public void NaiveBayesClassifier_CrossValidation(File TRAINING_DIR, File TESTING_DIR, String[] CATEGORIES, int NUM_FOLDS) throws IOException, ClassNotFoundException {
        XValidatingClassificationCorpus<CharSequence> corpus = new XValidatingClassificationCorpus<CharSequence>(NUM_FOLDS);
        System.out.println("Reading data.");
        // read data for train and test both

        for (int i = 0; i < CATEGORIES.length; ++i) {
            Classification c = new Classification(CATEGORIES[i]);
            File classDir = new File(TRAINING_DIR, CATEGORIES[i]);
            if (!classDir.isDirectory()) {
                String msg = "Could not find training directory=" + classDir + "\nHave you unpacked 4 newsgroups?";
                System.out.println(msg); // in case exception gets lost in shell
                throw new IllegalArgumentException(msg);
            }

            String[] trainingFiles = classDir.list();
            for (int j = 0; j < trainingFiles.length; ++j) {
                File fileF = new File(classDir, trainingFiles[j]);
                String text = Files.readFromFile(fileF, "ISO-8859-1");
                // System.out.println("Training on " + CATEGORIES[i] + "/" + trainingFiles[j]);
                corpus.handle(text, c);
            }
        }

        System.out.println("Num instances=" + corpus.numInstances() + ".");
        System.out.println("Permuting corpus.");
        corpus.permuteCorpus(new Random()); // destroys runs of categories

        for (int fold = 0; fold < NUM_FOLDS; ++fold) {
            corpus.setFold(fold);

            NaiveBayesClassifier classifier = new NaiveBayesClassifier(CATEGORIES, new IndoEuropeanTokenizerFactory());
            corpus.visitTrain(classifier);
            @SuppressWarnings("unchecked") // know type is ok by compilation
            Classifier<CharSequence, JointClassification> compiledClassifier = (Classifier<CharSequence, JointClassification>) AbstractExternalizable.compile(classifier);

            ClassifierEvaluator<CharSequence, JointClassification> evaluator = new ClassifierEvaluator<CharSequence, JointClassification>(compiledClassifier, CATEGORIES);
            corpus.visitTest(evaluator);
            System.out.println("toString:" + " " + evaluator.confusionMatrix().toString());
        }
    }

    //BernoulliClassifier
    public void BernoulliClassifier(File TRAINING_DIR, File TESTING_DIR, String[] CATEGORIES) throws IOException {
        FeatureExtractor FEATURE_EXTRACTOR = new TokenFeatureExtractor(IndoEuropeanTokenizerFactory.FACTORY);

        BernoulliClassifier classifier = new BernoulliClassifier(FEATURE_EXTRACTOR);

        //Training
        for (int i = 0; i < CATEGORIES.length; ++i) {
            File classDir = new File(TRAINING_DIR, CATEGORIES[i]);
            if (!classDir.isDirectory()) {
                String msg = "Could not find training directory=" + classDir + "\nHave you unpacked 4 newsgroups?";
                System.out.println(msg);
                throw new IllegalArgumentException(msg);
            }

            String[] trainingFiles = classDir.list();
            for (int j = 0; j < trainingFiles.length; ++j) {
                File fileF = new File(classDir, trainingFiles[j]);
                String text = Files.readFromFile(fileF, "ISO-8859-1");
                System.out.println("Training on " + CATEGORIES[i] + "/" + trainingFiles[j]);

                Classification classification = new Classification(CATEGORIES[i]);
                //El metodo handle entrena el clasificador en el objeto especificado(text) con la clasificacion(classification) especificada
                classifier.handle(text, classification);
            }
        }

        //Testing
        String bestCategory = "";
        File classDir = new File(TESTING_DIR.getAbsolutePath());
        String[] testingFiles = classDir.list();
        for (int j = 0; j < testingFiles.length; ++j) {
            String text = Files.readFromFile(new File(classDir, testingFiles[j]), "ISO-8859-1");
            System.out.print("Testing :" + "/" + testingFiles[j] + " ");

            JointClassification jc = classifier.classify(text);
            bestCategory = jc.bestCategory();
            System.out.println("Got best category of: " + bestCategory);
            String details = jc.toString();
            System.out.println(details);
            System.out.println("---------------");
        }
    }

    //BernoulliClassifier Cross Validation
    public void BernoulliClassifier_CrossValidation(File TRAINING_DIR, File TESTING_DIR, String[] CATEGORIES, int NUM_FOLDS) throws IOException {
        XValidatingClassificationCorpus<CharSequence> corpus = new XValidatingClassificationCorpus<CharSequence>(NUM_FOLDS);

        for (int i = 0; i < CATEGORIES.length; ++i) {
            Classification c = new Classification(CATEGORIES[i]);
            File classDir = new File(TRAINING_DIR, CATEGORIES[i]);
            if (!classDir.isDirectory()) {
                String msg = "Could not find training directory=" + classDir + "\nHave you unpacked 4 newsgroups?";
                System.out.println(msg);
                throw new IllegalArgumentException(msg);
            }

            String[] trainingFiles = classDir.list();
            for (int j = 0; j < trainingFiles.length; ++j) {
                File fileF = new File(classDir, trainingFiles[j]);
                String text = Files.readFromFile(fileF, "ISO-8859-1");
                corpus.handle(text, c);
            }
        }

        System.out.println("Num instances=" + corpus.numInstances() + ".");
        System.out.println("Permuting corpus.");
        corpus.permuteCorpus(new Random()); // destroys runs of categories

        for (int fold = 0; fold < NUM_FOLDS; ++fold) {
            corpus.setFold(fold);

            FeatureExtractor FEATURE_EXTRACTOR = new TokenFeatureExtractor(IndoEuropeanTokenizerFactory.FACTORY);
            BernoulliClassifier classifier = new BernoulliClassifier(FEATURE_EXTRACTOR);

            corpus.visitTrain(classifier);
            @SuppressWarnings("unchecked") // know type is ok by compilation
            //Classifier<CharSequence,JointClassification> compiledClassifier = (Classifier<CharSequence,JointClassification>)AbstractExternalizable.compile(classifier);
            ClassifierEvaluator<CharSequence, JointClassification> evaluator = new ClassifierEvaluator<CharSequence, JointClassification>(classifier, CATEGORIES);
            corpus.visitTest(evaluator);
            System.out.println("toString:" + " " + evaluator.confusionMatrix().toString());
        }
    }

    //K-NN Classifier
    public void K_NN_Classifier(File TRAINING_DIR, File TESTING_DIR, String[] CATEGORIES) throws IOException {
        TokenFeatureExtractor FEATURE_EXTRACTOR = new TokenFeatureExtractor(new IndoEuropeanTokenizerFactory());

        KnnClassifier<CharSequence> classifier = new KnnClassifier<CharSequence>(FEATURE_EXTRACTOR, 1);

        //Training
        for (int i = 0; i < CATEGORIES.length; ++i) {
            File classDir = new File(TRAINING_DIR, CATEGORIES[i]);
            if (!classDir.isDirectory()) {
                String msg = "Could not find training directory=" + classDir + "\nHave you unpacked 4 newsgroups?";
                System.out.println(msg);
                throw new IllegalArgumentException(msg);
            }

            String[] trainingFiles = classDir.list();
            for (int j = 0; j < trainingFiles.length; ++j) {
                File fileF = new File(classDir, trainingFiles[j]);
                String text = Files.readFromFile(fileF, "ISO-8859-1");
                System.out.println("Training on " + CATEGORIES[i] + "/" + trainingFiles[j]);

                Classification classification = new Classification(CATEGORIES[i]);
                //El metodo handle entrena el clasificador en el objeto especificado(text) con la clasificacion(classification) especificada
                classifier.handle(text, classification);
            }
        }

        //Testing
        String bestCategory = "";
        File classDir = new File(TESTING_DIR.getAbsolutePath());
        String[] testingFiles = classDir.list();
        for (int j = 0; j < testingFiles.length; ++j) {
            String text = Files.readFromFile(new File(classDir, testingFiles[j]), "ISO-8859-1");
            System.out.print("Testing :" + "/" + testingFiles[j] + " ");
            ScoredClassification jc = classifier.classify(text);
            bestCategory = jc.bestCategory();
            System.out.println("Got best category of: " + bestCategory);
            String details = jc.toString();
            System.out.println(details);
            System.out.println("---------------");
        }
    }

    //K-NN Classifier Cross Validation
    public void K_NN_Classifier_CrossValidation(File TRAINING_DIR, File TESTING_DIR, String[] CATEGORIES, int NUM_FOLDS) throws IOException {
        XValidatingClassificationCorpus<CharSequence> corpus = new XValidatingClassificationCorpus<CharSequence>(NUM_FOLDS);

        for (int i = 0; i < CATEGORIES.length; ++i) {
            Classification c = new Classification(CATEGORIES[i]);
            File classDir = new File(TRAINING_DIR, CATEGORIES[i]);
            if (!classDir.isDirectory()) {
                String msg = "Could not find training directory=" + classDir + "\nHave you unpacked 4 newsgroups?";
                System.out.println(msg);
                throw new IllegalArgumentException(msg);
            }

            String[] trainingFiles = classDir.list();
            for (int j = 0; j < trainingFiles.length; ++j) {
                File fileF = new File(classDir, trainingFiles[j]);
                String text = Files.readFromFile(fileF, "ISO-8859-1");
                corpus.handle(text, c);
            }
        }

        System.out.println("Num instances=" + corpus.numInstances() + ".");
        System.out.println("Permuting corpus.");
        corpus.permuteCorpus(new Random()); // destroys runs of categories

        for (int fold = 0; fold < NUM_FOLDS; ++fold) {
            corpus.setFold(fold);

            TokenFeatureExtractor FEATURE_EXTRACTOR = new TokenFeatureExtractor(new IndoEuropeanTokenizerFactory());
            KnnClassifier<CharSequence> classifier = new KnnClassifier<CharSequence>(FEATURE_EXTRACTOR, 3);

            corpus.visitTrain(classifier);
            @SuppressWarnings("unchecked") // know type is ok by compilation
            //Classifier<CharSequence,JointClassification> compiledClassifier = (Classifier<CharSequence,JointClassification>)AbstractExternalizable.compile(classifier);
            ClassifierEvaluator<CharSequence, ScoredClassification> evaluator = new ClassifierEvaluator<CharSequence, ScoredClassification>(classifier, CATEGORIES);
            corpus.visitTest(evaluator);
            System.out.println("toString:" + " " + evaluator.confusionMatrix().toString());
        }
    }

    //TradNaiveBayesClassifier
    public void TradNaiveBayesClassifier(File TRAINING_DIR, File TESTING_DIR, String[] CATEGORIES) throws IOException {
        TokenizerFactory TOKENIZER_FACTORY = IndoEuropeanTokenizerFactory.FACTORY;

        TradNaiveBayesClassifier classifier = new TradNaiveBayesClassifier((new HashSet<String>(Arrays.asList(CATEGORIES))), TOKENIZER_FACTORY, 1, 1, Double.NaN);

        //Training
        for (int i = 0; i < CATEGORIES.length; ++i) {
            File classDir = new File(TRAINING_DIR, CATEGORIES[i]);
            if (!classDir.isDirectory()) {
                String msg = "Could not find training directory=" + classDir + "\nHave you unpacked 4 newsgroups?";
                System.out.println(msg);
                throw new IllegalArgumentException(msg);
            }

            String[] trainingFiles = classDir.list();
            for (int j = 0; j < trainingFiles.length; ++j) {
                File fileF = new File(classDir, trainingFiles[j]);
                String text = Files.readFromFile(fileF, "ISO-8859-1");
                System.out.println("Training on " + CATEGORIES[i] + "/" + trainingFiles[j]);

                Classification classification = new Classification(CATEGORIES[i]);
                //El metodo handle entrena el clasificador en el objeto especificado(text) con la clasificacion(classification) especificada
                classifier.handle(text, classification);
            }
        }

        //Testing
        String bestCategory = "";
        File classDir = new File(TESTING_DIR.getAbsolutePath());
        String[] testingFiles = classDir.list();
        for (int j = 0; j < testingFiles.length; ++j) {
            String text = Files.readFromFile(new File(classDir, testingFiles[j]), "ISO-8859-1");
            System.out.print("Testing :" + "/" + testingFiles[j] + " ");
            JointClassification jc = classifier.classify(text);
            bestCategory = jc.bestCategory();
            System.out.println("Got best category of: " + bestCategory);
            String details = jc.toString();
            System.out.println(details);
            System.out.println("---------------");
        }
    }

    //TradNaiveBayesClassifier_CrossValidation
    public void TradNaiveBayesClassifier_CrossValidation(File TRAINING_DIR, File TESTING_DIR, String[] CATEGORIES, int NUM_FOLDS) throws IOException {
        XValidatingClassificationCorpus<CharSequence> corpus = new XValidatingClassificationCorpus<CharSequence>(NUM_FOLDS);

        for (int i = 0; i < CATEGORIES.length; ++i) {
            Classification c = new Classification(CATEGORIES[i]);
            File classDir = new File(TRAINING_DIR, CATEGORIES[i]);
            if (!classDir.isDirectory()) {
                String msg = "Could not find training directory=" + classDir + "\nHave you unpacked 4 newsgroups?";
                System.out.println(msg);
                throw new IllegalArgumentException(msg);
            }

            String[] trainingFiles = classDir.list();
            for (int j = 0; j < trainingFiles.length; ++j) {
                File fileF = new File(classDir, trainingFiles[j]);
                String text = Files.readFromFile(fileF, "ISO-8859-1");
                corpus.handle(text, c);
            }
        }

        System.out.println("Num instances=" + corpus.numInstances() + ".");
        System.out.println("Permuting corpus.");
        corpus.permuteCorpus(new Random()); // destroys runs of categories

        for (int fold = 0; fold < NUM_FOLDS; ++fold) {
            corpus.setFold(fold);

            TokenizerFactory TOKENIZER_FACTORY = IndoEuropeanTokenizerFactory.FACTORY;
            TradNaiveBayesClassifier classifier = new TradNaiveBayesClassifier((new HashSet<String>(Arrays.asList(CATEGORIES))), TOKENIZER_FACTORY, 7, 1, Double.NaN);

            corpus.visitTrain(classifier);
            @SuppressWarnings("unchecked") // know type is ok by compilation
            //Classifier<CharSequence,JointClassification> compiledClassifier = (Classifier<CharSequence,JointClassification>)AbstractExternalizable.compile(classifier);
            ClassifierEvaluator<CharSequence, JointClassification> evaluator = new ClassifierEvaluator<CharSequence, JointClassification>(classifier, CATEGORIES);

            corpus.visitTest(evaluator);
            System.out.println("toString:" + " " + evaluator.confusionMatrix().toString());
        }
    }
//    //Funcion que ordene(rank) teniendo en cuenta el score, fecha publicacion y tipo de FE
//    public void funcion(String[] CATEGORIES, ScoredClassification score, String FueExt, LinkedList<FuenteExterna> FuentesExternas) {//LinkedList fecha
//        //ponerlo en fichero
//        Date fecha = new Date(2010, 3, 24);
//        int date = fecha.getDate();
//        int day = fecha.getDay();
//        int month = fecha.getMonth();
//
//        System.out.println("date" + " " + date + "day" + " " + day + "month" + " " + month);
//
//        for (int i = 0; i < CATEGORIES.length; i++) {
//            double SCORE = score.score(i);
//            double FORMULA = 0;
//
//            for (int j = 0; j < FuentesExternas.size(); j++) {
//                if (FueExt.compareTo(FuentesExternas.get(i).tipo) == 0) {
//                    FORMULA = SCORE * (FuentesExternas.get(i).valor);
//                    System.out.println("score" + " " + FORMULA);
//                }
//            }
//        }
//    }
//------------------MAIN--------------------------------------------------------
//    public static void main(String[] args) {
//        try {
//            ProcesarNoticias claseMain = new ProcesarNoticias();
//            //------------------DECLARACIONES-----------------------------------------------
//            //Camino donde estan los ficheros .txt
//            File fileTest = new File("E:/Usuarios/Maria/TESIS TUTORA/XXXXXX/APLICACION/Directorio/data/fourNewsGroups/news-original-test/");
//            File fileTrain = new File("E:/Usuarios/Maria/TESIS TUTORA/XXXXXX/APLICACION/Directorio/data/fourNewsGroups/news-original-train/");
//            File TRAINING_DIR = new File("E:/Usuarios/Maria/TESIS TUTORA/XXXXXX/APLICACION/Directorio/data/fourNewsGroups/news-train/");
//            File TESTING_DIR = new File("E:/Usuarios/Maria/TESIS TUTORA/XXXXXX/APLICACION/Directorio/data/fourNewsGroups/news-test/");
//            String[] CATEGORIES = {"Cultura", "Deporte", "Especiales", "Holguin", "Internacional", "Nacional", "Opinion", "Salud"};
//            int NGRAM_SIZE = 6;
////            LinkedList<FuenteExterna> FuentesExternas = new LinkedList<FuenteExterna>();
////            FuentesExternas.add(new FuenteExterna("AIN", 5));
////            FuentesExternas.add(new FuenteExterna("Prensa Latina", 4));
////            FuentesExternas.add(new FuenteExterna("Cuba Debate", 3));
////            FuentesExternas.add(new FuenteExterna("Granma", 2));
////            FuentesExternas.add(new FuenteExterna("Juventud Rebelde", 1));
//            //------------------ACCESO A LOS METODOS-----------------------------------------------
//            /************************* PREPROCESAMIENTO DEL CCONJUNTO DE TEST ***********************************************************************/
//            //StandardAnalyzer, Stemming,elimino las StopWords para el conjunto de prueba o test
////            claseMain.preprocesarCorpus(fileTest.listFiles(), fileTest);
//            /*****************************************************************************************************************************************/
//            /************************* PREPROCESAMIENTO DEL CCONJUNTO DE TRAIN ***********************************************************************/
//            //StandardAnalyzer, Stemming,elimino las StopWords para el conjunto de prueba o test
////            claseMain.preprocesarCorpusEntrenamiento(fileTrain, CATEGORIES);
//            /*****************************************************************************************************************************************/
//            /************************* ENTRENAMIENTO Y CLASIFICACIONES ***********************************************************************/
//            //Clasificacion con DynamicLMClassifier con LingPipe
//            claseMain.clasificacionDynamicLMClassifier(TRAINING_DIR, TESTING_DIR, CATEGORIES, NGRAM_SIZE);
//            //claseMain.clasificacionDynamicLMClassifierCross_Validation(TRAINING_DIR, TESTING_DIR, CATEGORIES, NGRAM_SIZE, 5);
//            //Clasificacion TF_IDF con LingPipe
//            //claseMain.clasificacionTF_IDF(TRAINING_DIR, TESTING_DIR, CATEGORIES, FuentesExternas);
//            //claseMain.clasificacionTF_IDFCross_Validation(TRAINING_DIR, TESTING_DIR, CATEGORIES, FuentesExternas, 5);
//            //Clasificacion PerceptronClassifier (Binary Class) con LingPipe
//            //claseMain.clasificacionBinaryClassPerceptron(TRAINING_DIR, TESTING_DIR, CATEGORIES);
//            //Clasificacion LogisticRegressionClassifier (Multi Class) con LingPipe
//            //claseMain.clasificacionMultiClassRegresion(CATEGORIES, TRAINING_DIR, TESTING_DIR);
//            //claseMain.clasificacionMultiClassRegresionCross_Validation(CATEGORIES, TRAINING_DIR, TESTING_DIR, 5);
//            //Naive Bayes classifier
//            //claseMain.NaiveBayesClassifier(TRAINING_DIR, TESTING_DIR, CATEGORIES);
//            //claseMain.NaiveBayesClassifier_CrossValidation(TRAINING_DIR, TESTING_DIR, CATEGORIES, 10);
//            //BernoulliClassifier
//            //claseMain.BernoulliClassifier(TRAINING_DIR, TESTING_DIR, CATEGORIES);
//            //claseMain.BernoulliClassifier_CrossValidation(TRAINING_DIR, TESTING_DIR, CATEGORIES, 10);
//            //KNNClassifier
//            //claseMain.K_NN_Classifier(TRAINING_DIR, TESTING_DIR, CATEGORIES);
//            //claseMain.K_NN_Classifier_CrossValidation(TRAINING_DIR, TESTING_DIR, CATEGORIES, 10);
//            //TradNaiveBayesClassifier
//            //claseMain.TradNaiveBayes_Classifier(TRAINING_DIR, TESTING_DIR, CATEGORIES);
//            //claseMain.TradNaiveBayesClassifier_CrossValidation(TRAINING_DIR, TESTING_DIR, CATEGORIES, 5);
//            /*****************************************************************************************************************************************/
//        } catch (ClassNotFoundException ex) {
//            Logger.getLogger(ProcesarNoticias.class.getName()).log(Level.SEVERE, null, ex);
//        } catch (FileNotFoundException ex) {
//            Logger.getLogger(ProcesarNoticias.class.getName()).log(Level.SEVERE, null, ex);
//        } catch (IOException ex) {
//            Logger.getLogger(ProcesarNoticias.class.getName()).log(Level.SEVERE, null, ex);
//        }
//
//    }
//------------------FIN DEL MAIN
}

class TestFeatureExtractor implements FeatureExtractor<CharSequence>, Serializable {

    public Map<String, Double> features(CharSequence in) {
        ObjectToDoubleMap<String> map = new ObjectToDoubleMap<String>();
        char[] cs = Strings.toCharArray(in);
        Tokenizer tokenizer = IndoEuropeanTokenizerFactory.FACTORY.tokenizer(cs, 0, cs.length);
        for (String token : tokenizer) {
            map.increment(token, 1.0);
        }
        //  System.out.println(in + "=" + map);
        return map;
    }
}

class TestCorpus extends Corpus<ClassificationHandler<CharSequence, Classification>> {

    final File mTrainFile;
    final String[] mCats;

    TestCorpus(File TrainFile, String[] cats) {
        mTrainFile = TrainFile;
        mCats = cats;
    }

    @Override
    public void visitTrain(ClassificationHandler<CharSequence, Classification> handler) throws IOException {
        for (int i = 0; i < mCats.length; ++i) {
            File classDir = new File(mTrainFile, mCats[i]);
            if (!classDir.isDirectory()) {
                String msg = "Could not find training directory=" + classDir + "\nHave you unpacked 4 newsgroups?";
                System.out.println(msg); // in case exception gets lost in shell
                throw new IllegalArgumentException(msg);
            }

            String[] trainingFiles = classDir.list();
            for (int j = 0; j < trainingFiles.length; ++j) {
                File fileF = new File(classDir, trainingFiles[j]);
                String text = Files.readFromFile(fileF, "ISO-8859-1");
                System.out.println("Training on " + mCats[i] + "/" + trainingFiles[j]);
                Classification c = new Classification(mCats[i]);
                handler.handle(text, c);
            }
        }
    }
}
